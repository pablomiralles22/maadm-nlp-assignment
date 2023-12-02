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
```bash
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
    --hf-env-file .env \
    --hf-repository maadm-nlp-group-b/maadm-nlp-pan23-task1-roberta-base-finetuned
```

## Methodology

### Dataset
We use the PAN23 multi-author analysis dataset. The dataset is transformed to a binary classification task using the script `pan-data-transform`:
```bash
python scripts/pan-data-transform.py --source-dir data/pan23/original --target-dir data/pan23/transformed --augment
```
#### Data augmentation
The `--augment` flag in the previous command augments the training data in the following way.

Let $p_1, \dots, p_n$ be the paragraphs of a document, with labels $l_1, \dots, l_{n-1}$. Now, if $l_1=l_2=0$, we know that $p_1, p_2, p_3$ are written by the same author. Thus, as long as we find contiguous $l_i=0$, we can group the corresponding paragraphs together. We build groups of paragraphs $g_1,\dots, g_m$ using this procedure, were each $p, q \in g_i$ are of the same author, and each $p \in g_i, q \in g_{i+1}$ belong to different authors. This way, we get many more training examples than we originally had.

### Models
- Fine-tuned `roberta-base`.
- Ensemble of the previous fine-tuned models and a convolutional model.

### Training procedure
- Training on 16-bit mixed precision.
- Weighted cross entropy for class imbalances (in particular task 1).

```bash
# for the transformer
conda run -n master-nlp python scripts/train.py --config configs/roberta-base.json

# for the ensemble (run after uploading to hugginface)
conda run -n master-nlp python scripts/train.py \
    -c configs/roberta-base-ensemble-task1.json  \
       configs/roberta-base-ensemble-task2.json  \
       configs/roberta-base-ensemble-task3.json
```

## Evaluation results

- `roberta-base`.
    - **Task 1**: 0.9807 F1
    - **Task 2**: 0.7657 F1
    - **Task 3**: 0.6668 F1
- `ensemble` with `conv`.
    - **Task 1**: 0.984 F1
    - **Task 2**: 0.770 F1
    - **Task 3**: 0.677 F1

## Reproducing the results
If someone wants to reproduce the reults, we have uploaded the `out` directory with the training logs and the best checkpoints ([download link](https://upm365-my.sharepoint.com/:u:/g/personal/pablo_miralles_upm_es/Ef_zO0uBExVJmu6-R9Q9sFIB5l2_V6NUpbIk27TfBNQrow?e=opGHS9)). It should be downloaded and placed in the root directory of the project. Now, we will go over each step of our procedure.

### 1. Train the transformer model
We ran the following command, producing the `out/roberta-base` directory.
```bash
conda run -n master-nlp python scripts/train.py --config configs/roberta-base.json
```
### 2. Evaluate the fine-tuned transformer models
```bash
conda run -n master-nlp python scripts/evaluate.py \
    --model-config configs/roberta-base.json \
    --checkpoint out/roberta-base/finetuned/task1/lightning_logs/version_1/checkpoints/epoch=8-val_f1_score=0.99.ckpt \
    --task 1 \
    --source-data-dir data/release

conda run -n master-nlp python scripts/evaluate.py \
    --model-config configs/roberta-base.json \
    --checkpoint out/roberta-base/finetuned/task2/lightning_logs/version_1/checkpoints/epoch=6-val_f1_score=0.76.ckpt \
    --task 2 \
    --source-data-dir data/release

conda run -n master-nlp python scripts/evaluate.py \
    --model-config configs/roberta-base.json \
    --checkpoint out/roberta-base/finetuned/task3/lightning_logs/version_1/checkpoints/epoch=3-val_f1_score=0.69.ckpt \
    --task 3 \
    --source-data-dir data/release
```
Notice that `data/release` should be the path to the original data.

### 3. Upload the fine-tuned transformer models
```bash
conda run -n master-nlp python scripts/upload_pretrained_transformer.py \
    --model-config configs/roberta-base.json \
    --checkpoint out/roberta-base/finetuned/task1/lightning_logs/version_1/checkpoints/epoch=8-val_f1_score=0.99.ckpt \
    --task 1 \
    --hf-env-file .env \
    --hf-repository maadm-nlp-group-b/maadm-nlp-pan23-task1-roberta-base-finetuned


conda run -n master-nlp python scripts/upload_pretrained_transformer.py \
    --model-config configs/roberta-base.json \
    --checkpoint out/roberta-base/finetuned/task2/lightning_logs/version_1/checkpoints/epoch=6-val_f1_score=0.76.ckpt \
    --task 2 \
    --hf-env-file .env \
    --hf-repository maadm-nlp-group-b/maadm-nlp-pan23-task2-roberta-base-finetuned


conda run -n master-nlp python scripts/upload_pretrained_transformer.py \
    --model-config configs/roberta-base.json \
    --checkpoint out/roberta-base/finetuned/task3/lightning_logs/version_1/checkpoints/epoch=3-val_f1_score=0.69.ckpt \
    --task 3 \
    --hf-env-file .env \
    --hf-repository maadm-nlp-group-b/maadm-nlp-pan23-task3-roberta-base-finetuned
```

### 4. Train the ensemble models
```bash
conda run -n master-nlp python scripts/train.py \
    -c configs/roberta-base-ensemble-task1.json  \
       configs/roberta-base-ensemble-task2.json  \
       configs/roberta-base-ensemble-task3.json
```

### 5. Evaluate the ensemble models
```bash
conda run -n master-nlp python scripts/evaluate.py \
    --model-config configs/roberta-base-ensemble-task1.json \
    --checkpoint out/roberta-base-ensemble/finetuned/task1/lightning_logs/version_1/checkpoints/epoch=2-val_f1_score=0.99.ckpt \
    --task 1 \
    --source-data-dir data/release

conda run -n master-nlp python scripts/evaluate.py \
    --model-config configs/roberta-base-ensemble-task2.json \
    --checkpoint out/roberta-base-ensemble/finetuned/task2/lightning_logs/version_0/checkpoints/epoch=0-val_f1_score=0.76.ckpt \
    --task 2 \
    --source-data-dir data/release

conda run -n master-nlp python scripts/evaluate.py \
    --model-config configs/roberta-base-ensemble-task3.json \
    --checkpoint out/roberta-base-ensemble/finetuned/task3/lightning_logs/version_0/checkpoints/epoch=1-val_f1_score=0.69.ckpt \
    --task 3 \
    --source-data-dir data/release
```

Notice again that `data/release` should be the path to the original data.