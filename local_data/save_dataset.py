import argparse
import os
from datasets import load_dataset


# def main():
#     ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="val")
#     ds.save_to_disk("local_data/data/wikitext-2-raw-v1") 

def main():
    parser = argparse.ArgumentParser(description="Save HuggingFace dataset locally.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., Salesforce/wikitext)')
    parser.add_argument('--config', type=str, required=True, help='Dataset config (e.g., wikitext-2-raw-v1)')
    parser.add_argument('--split', type=str, required=True, help='Dataset split (e.g., validation)')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save dataset (relative to project root)')
    args = parser.parse_args()

    ds = load_dataset(args.dataset_name, args.config, split=args.split)
    save_path = os.path.abspath(args.save_path)
    os.makedirs(save_path, exist_ok=True)
    ds.save_to_disk(save_path)

if __name__ == '__main__':
    main()
