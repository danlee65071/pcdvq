import argparse
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import gc
from tqdm.auto import tqdm
from contextlib import contextmanager

from pcdvq import (
    e8_minimal_directions,
    construct_direction_codebook,
    construct_magnitude_codebook,
    PCDVQ,
)
from pcdvq.utils import reshape_pq_to_k, reshape_k_to_pq


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


distortion_rate = {}


torch.set_grad_enabled(False)
torch.set_num_threads(max(1, torch.get_num_threads()))


@contextmanager
def inference_ctx():
    with torch.inference_mode():
        yield

def quantize_linear_inplace(
    module: nn.Module,
    k: int,
    phi_bits: int,
    r_bits: int,
    tau: float = 0.99,
    tol: float = 1e-3,
    iters: int = 100,
):
    global distortion_rate

    linears: list[tuple[str, nn.Linear]] = [
        (name, m) for name, m in module.named_modules() if isinstance(m, nn.Linear)
    ]

    e8_dirs = e8_minimal_directions()
    C_phi = construct_direction_codebook(e8_dirs, phi_bits)
    C_r = construct_magnitude_codebook(r_bits, k, tau, tol, iters)
    pcdvq = PCDVQ(directions_codebook=C_phi, magnitudes_codebook=C_r)

    progress = tqdm(total=len(linears))
    with inference_ctx():
        for name, lin in linears:
            w_dev = lin.weight.device
            w_dtype = lin.weight.dtype

            W_cpu = lin.weight.detach().to("cpu", copy=True)
            p, q  = W_cpu.shape

            Y  = reshape_pq_to_k(W_cpu, k)
            Z  = pcdvq.forward(Y)["x_q"]
            Wq_cpu = reshape_k_to_pq(Z, p, q)

            mse_val = F.mse_loss(W_cpu.float(), Wq_cpu.float()).item()
            distortion_rate[name] = mse_val

            if torch.cuda.is_available() and w_dev.type == "cuda":
                Wq_cpu = Wq_cpu.contiguous().pin_memory()
                lin.weight.data.copy_(Wq_cpu.to(device=w_dev, dtype=w_dtype, non_blocking=True))
            else:
                lin.weight.data.copy_(Wq_cpu.to(dtype=w_dtype))

            del W_cpu, Y, Z, Wq_cpu
            progress.update(1)

    progress.close()


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
    parser.add_argument("--max_length", type=int, default=None,
                        help="window stride")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="enable if the model requires custom code")
    parser.add_argument("--quantize_with_pcdvq", action="store_true",
                        help="enable PCDVQ quantization of linear layers")
    parser.add_argument("--k", type=int, default=256, help="k")
    parser.add_argument("--phi_bits", type=int, default=5,
                        help="bits for phi")
    parser.add_argument("--r_bits", type=int, default=3,
                        help="bits for r")
    parser.add_argument("--tau", type=float, default=0.9,
                        help="batch size")
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="batch size")
    parser.add_argument("--save_path", type=str, default=None,
                        help="save path for quantized model")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size")
    parser.add_argument("--device", type=str, default='cpu',
                        help="device")
    parser.add_argument("--use_cache", action="store_true",
                        help="use KV cache")
    args = parser.parse_args()
    
    dtype = torch.float16
    device = args.device
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=False
    )

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    texts = [text for text in dataset["text"] if isinstance(text, str) and text.strip() != ""]
    
    save_path = args.model_name
    if args.quantize_with_pcdvq:
        model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=args.trust_remote_code,
    )
        model.eval()
        model.config.use_cache = True if args.use_cache else False

        logger.info("Quantizing linear layers with PCDVQ...")
        quantize_linear_inplace(model, k=args.k, phi_bits=args.phi_bits,
                                r_bits=args.r_bits, tau=args.tau, tol=args.tol)
        logger.info("Quantization done.")
        
        if args.save_path is None:
            project_path = Path.cwd()
            model_dir = Path("quant_models", args.model_name)
            save_path = project_path / model_dir
            save_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    ppl = evaluate.load("evaluate/metrics/perplexity/perplexity.py")
    results = ppl.compute(
        model_id=save_path,
        batch_size=args.batch_size,
        device=device,
        add_start_token=False,
        max_length=args.max_length,
        predictions=texts[: args.batch_size * 20]
    )
    print(results)
    print(distortion_rate)
    with open('distortion_rate.json', "w") as json_file:
        json.dump(distortion_rate, json_file, indent=4)

if __name__ == '__main__':
    main()
