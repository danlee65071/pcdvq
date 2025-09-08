# Local Data Scripts

## Save a HuggingFace Model

From the project root, run:

```
python local_data/save_model.py --model_name <MODEL_NAME_OR_PATH> --save_path <LOCAL_SAVE_PATH>
```

Example:
```
python local_data/save_model.py --model_name Qwen/Qwen-7B --save_path local_data/data/qwen-7b
```

## Save a HuggingFace Dataset

From the project root, run:

```
python local_data/save_dataset.py --dataset_name <DATASET_NAME> --config <CONFIG> --split <SPLIT> --save_path <LOCAL_SAVE_PATH>
```

Example:
```
python local_data/save_dataset.py --dataset_name Salesforce/wikitext --config wikitext-2-raw-v1 --split validation --save_path local_data/data/wikitext-2-raw-v1
```
