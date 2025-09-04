import math
import functools
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate


from polar_decoupling import PCDVQ
from codebooks import (
    e8_minimal_directions,
    construct_direction_codebook,
    construct_magnitude_codebook,
)


import requests
from huggingface_hub import configure_http_backend

# def backend_factory():
#     s = requests.Session()
#     s.proxies = {
#         "http":  "",
#         "https": "",
#     }
#     # # if using a corporate CA:
#     # s.verify = "/path/to/company-ca.pem"
#     return s

# configure_http_backend(backend_factory)

from huggingface_hub import HfApi




class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        out = self.linear(x)
        return out


def reshape_pq_to_k(x: torch.Tensor, k: int, pad_value=0):
    """
    Reshape a (p, q) tensor to shape (ceil(p*q / k), k).
    If p*q % k != 0, pad at the end with `pad_value`.

    Args:
        x: 2D tensor of shape (p, q)
        k: positive integer chunk size for the last dimension
        pad_value: scalar used for padding

    Returns:
        y: tensor of shape (ceil(p*q / k), k)
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (p, q), got {x.ndim}D")

    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive int, got {k}")

    p, q = x.shape
    n = p * q
    rem = n % k

    flat = x.reshape(-1)
    if rem != 0:
        pad_elems = k - rem
        pad = flat.new_full((pad_elems,), pad_value)
        flat = torch.cat([flat, pad], dim=0)

    y = flat.view(-1, k)
    return y


def reshape_k_to_pq(y: torch.Tensor, p: int, q: int):
    """
    Inverse of reshape_pq_to_k. Assumes y is shape (ceil(p*q/k), k).
    Trims any padding at the end and reshapes to (p, q).
    """
    if y.ndim != 2:
        raise ValueError(f"y must be 2D, got {y.ndim}D")
    n = p * q
    if n > y.numel():
        raise ValueError(f"Target size {n} exceeds available elements {y.numel()}.")

    flat = y.reshape(-1)          # same order as before
    x = flat[:n].reshape(p, q)    # drop padding, restore shape
    return x


def replace_linear_with_custom(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None

            new_layer = CustomLinear(in_features, out_features, bias)

            k = 9
            original_weights_data = child.weight.data
            p, q = original_weights_data.shape
            reshaped_original_weight = reshape_pq_to_k(original_weights_data, k)
            phi_bits = 5
            r_bits = 3
            e8_dirs = e8_minimal_directions()
            C_phi = construct_direction_codebook(e8_dirs, phi_bits)
            C_r = construct_magnitude_codebook(r_bits, k, 0.99, 1e-3, 100)
            pcdvq = PCDVQ(
                directions_codebook=C_phi,
                magnitudes_codebook=C_r,
            )
            dict_pcdvq = pcdvq.forward(reshaped_original_weight)
            quant_weight = dict_pcdvq['x_q']
            quant_weight = reshape_k_to_pq(quant_weight, p, q)

            new_layer.linear.weight.data.copy_(quant_weight)
            if bias:
                new_layer.linear.bias.data.copy_(child.bias.data)

            setattr(module, name, new_layer)
        else:
            replace_linear_with_custom(child)

def tokenize_fn(example):
    return tokenizer(example["text"], return_attention_mask=False)

def group_texts(examples, block_size=2048):
    # Concatenate all texts
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result

def main():
 
    
    model_id = "meta-llama/Meta-Llama-3-8B"
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="eval")
    block_size = 2048

    
#     HF_ENDPOINT = 'http://mirrors.tools.huawei.com/huggingface'
#     api = HfApi(endpoint=HF_ENDPOINT)
#     api.snapshot_download(
#         repo_id=model_id,
#         repo_type="model",
#         revision="main",
#         local_dir="./",
#         etag_timeout=10000
#     )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
    )
    replace_linear_with_custom(model)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    group_texts_with_block = functools.partial(group_texts, block_size=block_size)
    lm_dataset = tokenized.map(group_texts_with_block, batched=True)

    perplexity = evaluate.load("perplexity", module_type="metric")

    results = perplexity.compute(
        model_id=model_id,
        add_start_token=False,
        predictions=lm_dataset["input_ids"],
        tokenizer=tokenizer,
        device="cuda",
    )

    print("Perplexity:", results["perplexity"])

if __name__ == '__main__':
    main()
