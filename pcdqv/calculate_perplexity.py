import evaluate
import math
import functools
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


from quant_linear import QuantLinear


def replace_linear_with_custom(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            bias = child.bias is not None

            new_layer = QuantLinear(in_features, out_features, bias)

            new_layer.linear.weight.data.copy_(child.weight.data)
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
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    block_size = 2048

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
