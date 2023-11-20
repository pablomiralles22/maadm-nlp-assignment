import time
import numpy as np
import torch

from torch.utils.data.dataloader import DataLoader
from losses.ntxent import ntxent_loss

class ContrastivePretrainer:
    """
    Given a model that projects each example to a one-dimensional embedding
    through a projection head, trains the model through contrastive learning.
    The data loader is assumed to give a single batch, where each positive pair
    is concatenated.
    """
    @classmethod
    def get_default_config(cls) -> dict:
        return {
            "num_workers": 4,
            "steps": None,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "betas": (0.9, 0.99),
            "weight_decay": 0.,
            "log_file": None,
            "checkpoint_file": None,
            "eval_every": 100,
            "collator_fn": None,
            "device": "cpu",
        }

    def __init__(self, config, model, train_dataset, test_dataset):
        self.config = {
            **self.get_default_config(),
            **config,
        }

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def run(self):
        model, config = self.model, self.config

        model = model.to(config["device"])

        # setup the optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            betas=config["betas"],
            weight_decay=config["weight_decay"],
        )

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            collate_fn=config["collator_fn"],
        )
        test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            collate_fn=config["collator_fn"],
        )

        # set training mode for model
        model.train()

        # initialize training loop
        iter_num = 0
        iter_time = time.time()
        max_iters = config.get("steps")

        log_file = config.get("log_file")
        
        losses = []
        best_eval_loss = 1e10  # INF

        data_iter = iter(train_loader)
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            loss = self.__loss_step(batch, model, optimizer)
            losses.append(loss)

            iter_num += 1

            # log and store checkpoint
            if iter_num % config["eval_every"] == 0:
                mean_training_loss = np.mean(losses)
                mean_eval_loss = self.__mean_eval_loss(model, test_loader)

                tnow = time.time()
                iters_dt = tnow - iter_time
                iter_time = tnow

                self.__print_and_log(
                    f"TRAIN - {iter_num=}, {mean_training_loss=}, {mean_eval_loss=}, ({iters_dt:.2f}s)",
                    log_file,
                )

                if mean_eval_loss < best_eval_loss:
                    best_eval_loss = mean_eval_loss
                    torch.save(model, config["checkpoint_file"])

                losses = []
            
            # termination conditions
            if max_iters is not None and iter_num >= max_iters:
                break

    def __loss_step(self, batch, model, optimizer):
        input_ids = batch["input_ids"].to(self.config["device"])
        attention_mask = batch["attention_mask"].to(self.config["device"])

        # get model output
        emb = model(input_ids, attention_mask)
        emb_1, emb_2 = emb[::2], emb[1::2]
        loss = ntxent_loss(emb_1, emb_2, 0.07, self.config["device"])

        # backprop and update the parameters
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    def __mean_eval_loss(self, model, data_loader):
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.config["device"])
                attention_mask = batch["attention_mask"].to(self.config["device"])

                # get model output
                emb = model(input_ids, attention_mask)
                emb_1, emb_2 = emb[::2], emb[1::2]
                loss = ntxent_loss(emb_1, emb_2, 0.07, self.config["device"])
                losses.append(loss.item())
        model.train()
        return np.mean(losses)

    def __print_and_log(self, message, log_file):
        print(message)
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")