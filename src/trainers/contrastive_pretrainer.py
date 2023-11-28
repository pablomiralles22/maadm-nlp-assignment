import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from losses.ntxent import ntxent_loss
from typing import Sequence


class ContrastivePretrainingModule(pl.LightningModule):
    @classmethod
    def get_default_optimizer_config(cls) -> dict:
        return {
            "lr": 1e-4,
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
        }

    def __init__(
        self,
        model,  # Model with projection head
        optimizer_config,
    ):
        super().__init__()
        self.optimizer_config = {
            **self.get_default_optimizer_config(),
            **optimizer_config,
        }
        self.model = model

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log_dict({"train_loss": loss}, on_epoch=True, on_step=False, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log_dict({"val_loss": loss}, on_epoch=True, on_step=False, prog_bar=True)
        return {"loss": loss}

    def _step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # get model output
        emb = self.model(input_ids, attention_mask)
        emb_1, emb_2 = emb[::2], emb[1::2]
        return ntxent_loss(emb_1, emb_2)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(), **self.optimizer_config
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
            },
        }
    
    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        return super().configure_callbacks() + [
            ModelCheckpoint(
                filename="{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
        ]
