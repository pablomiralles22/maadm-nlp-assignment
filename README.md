# NLP Assignment

## Usage
To train, run the `train.py` script with some config file, similar to `configs/base-config.json`.
```
python scripts/train.py --config configs/base-config.json
```

## Datasets
- PAN23 multi-author analysis. The dataset is transformed into a binary classification task using the script `pan-data-transform` (e.g. `python scripts/pan-data-transform.py --source-dir data/pan23/original --target-dir data`).
- Curated version of the [Blog Authorship corpus](https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm). Each directory refers to a single author. Posts shorter than 200 characters have been removed.

## Model
Fine-tuned `roberta-base`.

## Main ideas
- Fine-tuned on 16-bit mixed precision.
- Data augmentation.
- Weighted cross entropy for unbalances (in particular task 1).

## Training
- `roberta-base`.
    - **Task 1**
       - 10 epochs, 6 layers, `out/roberta-base/finetuned/task1/lightning_logs/version_12/checkpoints`
       - 20 epochs, 8 layers, `out/roberta-base/finetuned/task1/lightning_logs/version_12/checkpoints`
    - **Task 2**
       - 10 epochs, 6 layers, `out/roberta-base/finetuned/task1/lightning_logs/version_12/checkpoints`
       - 9 epochs, 8 layers, `out/roberta-base/finetuned/task1/lightning_logs/version_12/checkpoints`
    - **Task 3**
       - 10 epochs, 6 layers, `out/roberta-base/finetuned/task1/lightning_logs/version_12/checkpoints`
       - 9 epochs, 8 layers, `out/roberta-base/finetuned/task1/lightning_logs/version_12/checkpoints`