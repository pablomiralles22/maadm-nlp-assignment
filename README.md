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
Combination of `roberta-base` and a smaller convolutional model. The hypothesis is that stylistic features are mainly local, and convolutional layers have a good inductive bias for that.