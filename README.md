# Deep-Learning-for-Computer-Vision-Assignment-1

This project supports grader-friendly execution by using a model configuration file for training and storing the required metadata inside the saved checkpoint.

## What is the model configuration file?

`model_config.json` is a JSON file that stores model/training settings:

- `num_classes` (auto-updated from dataset folders during training)
- `epochs` (optional, default `50`)
- `batch_size` (optional, default `32`)
- `lr` (optional, default `0.05`)

Example:

```json
{
  "num_classes": 10,
  "epochs": 50,
  "batch_size": 32,
  "lr": 0.05
}
```

## Train

During training, `num_classes` is computed automatically from the number of class subfolders inside `--train_path` and written back into `model_config.json`.

```bash
python train.py \
  --train_path /path/to/train_dataset \
  --config_path model_config.json \
  --save_path model_weights.pkl
```

Training saves:
- model weights
- essential metadata (`num_classes`, `batch_size`) inside the same checkpoint file

## Evaluate

For checkpoints produced by the updated `train.py`, grader only needs dataset path + saved weights:

```bash
python eval.py \
  --test_path /path/to/hidden_test_parent_dir \
  --weights_path model_weights.pkl
```
