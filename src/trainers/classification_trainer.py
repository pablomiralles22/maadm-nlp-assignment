import torch
import pytorch_lightning as pl
import torchmetrics

from torch import nn


class ClassificationModule(pl.LightningModule):
    @classmethod
    def get_default_optimizer_config(cls) -> dict:
        return {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
        }

    def __init__(
        self,
        model,
        optimizer_config,
        positive_ratio,  # ratio of positive samples in the dataset, to correct imbalance
    ):
        super().__init__()
        self.optimizer_config = {
            **self.get_default_optimizer_config(),
            **optimizer_config,
        }
        self.model = model
        self.f1_score = torchmetrics.F1Score(task="binary")
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_ratio))

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch, batch_idx)
        f1_score = self.f1_score(logits, labels)
        self.log_dict(
            {
                "train_loss": loss,
                "train_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "logits": logits, "labels": labels}

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._step(batch, batch_idx)
        f1_score = self.f1_score(logits, labels)
        self.log_dict(
            {
                "val_loss": loss,
                "val_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "logits": logits, "labels": labels}

    def _step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].float().view(-1, 1)

        logits = self.model(input_ids, attention_mask)  # (BATCH_SIZE, 1)
        loss = self.loss_fn(logits, labels)
        return loss, logits, labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), **self.optimizer_config
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_f1_score",
                "interval": "epoch",
            },
        }
