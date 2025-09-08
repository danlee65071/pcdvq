import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


# def main():
#     model_name = "Qwen/Qwen-7B"
#
#     model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype="auto")
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#
#     save_path = "local_data/qwen-7b"
#     model.save_pretrained(save_path)
#     tokenizer.save_pretrained(save_path)

def main():
    parser = argparse.ArgumentParser(description="Save HuggingFace model and tokenizer locally.")
    parser.add_argument('--model_name', type=str, required=True, help='Model name or path (e.g., Qwen/Qwen-7B)')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save model and tokenizer (relative to project root)')
    args = parser.parse_args()

    # Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Ensure save_path is relative to project root
    save_path = os.path.abspath(args.save_path)
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == '__main__':
    main()
