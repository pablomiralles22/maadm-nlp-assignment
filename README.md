# NLP Assignment

## Usage

First, if you want to use one of our private pretrained models, you will need to fill in your Huggingface read and write keys in `example.env` and move it to `.env`.

Second, you will need to download the PAN23 dataset ([link](https://zenodo.org/records/7729178/files/pan23-multi-author-analysis.zip?download=1)) and unzip it in your workspace. Then, you will need to use our script to transform the data, and use the `-a` flag for augmentation:

```bash
micromamba run -n master-nlp python scripts/pan-data-transform.py -s data/release -t data/pan23/transformed -a
```

Finally, you can build a python environment with `conda` via the `conda-env.yml` file.

### Training
To train, run the `train.py` script with some config file, similar to `configs/base-config.json`.
```
python scripts/train.py --config configs/roberta-base.json
```

### Evaluating a model
The evaluation in the rest of the papers is done as the average of the F1-scores for each document. However, for training we transformed each pair of paragraphs as an individual example. Thus, we prepared another script to evaluate a model in a fair way.

```bash
python scripts/evaluate.py \
    --model-config configs/roberta-base.json \
    --checkpoint out/roberta-base/finetuned/task1/lightning_logs/version_16/checkpoints/epoch=9-val_f1_score=0.99.ckpt \
    --task 1 \
    --source-data-dir data/pan23/original
```

### Uploading a transformer model after fine-tuning
We have prepared a script to upload fine-tuned transformer model to Huggingface, from a given Pytorch Lightning checkpoint.

```bash
python scripts/upload_pretrained_transformer.py \
    --model-config configs/roberta-base-task1.json \
    --checkpoint out/roberta-base/finetuned/task1/lightning_logs/version_13/checkpoints/epoch=14-val_f1_score=0.99.ckpt \
    --task 1 \
    --hf-token <YOUR_TOKEN> \
    --hf-repository maadm-nlp-group-b/maadm-nlp-pan23-task1-roberta-base-finetuned
```

## Methodology and results

### Dataset
We use the PAN23 multi-author analysis dataset. The dataset is transformed to a binary classification task using the script `pan-data-transform`:
```bash
python scripts/pan-data-transform.py --source-dir data/pan23/original --target-dir data/pan23/transformed
```
#### Data augmentation
Further, you can add the `--augment` flag to the previous command, and the training sets will be augmented in the following way.

Let $p_1, \dots, p_n$ be the paragraphs of a document, with labels $l_1, \dots, l_{n-1}$. Now, if $l_1=l_2=0$, we know that $p_1, p_2, p_3$ are written by the same author. Thus, as long as we find contiguous $l_i=0$, we can group the corresponding paragraphs together. We build groups of paragraphs $g_1,\dots, g_m$ using this procedure, were each $p, q \in g_i$ are of the same author, and each $p \in g_i, q \in g_{i+1}$ belong to different authors. This way, we get many more training examples than we originally had.

### Models
- Fine-tuned `roberta-base`.
- Ensemble of the previous fine-tuned models and a convolutional model.

### Training procedure
- Training on 16-bit mixed precision.
- Weighted cross entropy for class imbalances (in particular task 1).

- `roberta-base`. Fine-tuned using file `configs/roberta-base.json`.
- `ensemble`. Fine-tuned using files... TODO

### Evaluation results

- `roberta-base`.
    - **Task 1**: 0.9807 F1
    - **Task 2**: 0.7657 F1
    - **Task 3**: 0.6668 F1
- `ensemble` with `conv`.
    - **Task 1**
    - **Task 2**
    - **Task 3**

