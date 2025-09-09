import argparse
from pathlib import Path
import logging
import torch
import torch.nn as nn
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from pcdvq import (
    e8_minimal_directions,
    construct_direction_codebook,
    construct_magnitude_codebook,
    PCDVQ,
)
from pcdvq.utils import reshape_pq_to_k, reshape_k_to_pq


def quantize_linear_inplace(module, k, phi_bits=5, r_bits=3,
                            tau=0.99, tol=1e-3, iters=100):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            with torch.no_grad():
                W = child.weight.data.detach().to("cpu")
                p, q = W.shape
                Y = reshape_pq_to_k(W, k)
                e8_dirs = e8_minimal_directions()
                C_phi = construct_direction_codebook(e8_dirs, phi_bits)
                C_r = construct_magnitude_codebook(r_bits, k, tau, tol, iters)
                pcdvq = PCDVQ(directions_codebook=C_phi, magnitudes_codebook=C_r)
                Wq = pcdvq.forward(Y)["x_q"]
                Wq = reshape_k_to_pq(Wq, p, q).to(device=child.weight.device, dtype=child.weight.dtype)
                child.weight.data.copy_(Wq)
        else:
            quantize_linear_inplace(child, k=k, phi_bits=phi_bits, r_bits=r_bits,
                                    tau=tau, tol=tol, iters=iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                        help="model id")
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="dataset configuration name (e.g. wikitext-2-raw-v1)")
    parser.add_argument("--split", type=str, default="validation",
                        help="dataset split")
    parser.add_argument("--stride", type=int, default=None,
                        help="window stride")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="enable if the model requires custom code")
    parser.add_argument("--quantize_with_pcdvq", action="store_true",
                        help="enable PCDVQ quantization of linear layers")
    parser.add_argument("--k", type=int, default=256, help="k")
    parser.add_argument("--save_path", type=str, default=None,
                        help="save path for quantized model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size")
    parser.add_argument("--device", type=str, default='cpu',
                        help="device")
    args = parser.parse_args()
    
    dtype = torch.float16
    device = args.device
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    texts = [text for text in dataset["text"] if isinstance(text, str) and text.strip() != ""]
    
    save_path = args.model_name
    if args.quantize_with_pcdvq:
        logger.info("Quantizing linear layers with PCDVQ...")
        quantize_linear_inplace(model, k=args.k)
        logger.info("Quantization done.")
        
        if args.save_path is None:
            project_path = Path.cwd()
            model_dir = Path("quant_models", args.model_name)
            save_path = project_path / model_dir
            save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)    
    ppl = evaluate.load("perplexity")
    results = ppl.compute(
        model_id=save_path,
        batch_size=args.batch_size,
        device=device,
        add_start_token=False,
        max_length=4096,
        predictions=texts[:8*8]
    )
    print(results)


if __name__ == '__main__':
    main()
