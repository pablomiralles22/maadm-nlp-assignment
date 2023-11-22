import time
import numpy as np
import torch

from torch import nn
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import confusion_matrix


class ClassificationTrainer:
    """
    Given a model with a classification head, trains the model.
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
            "collator_fn": None,
            "eval_every": 200,
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

        # create loss
        n_positives, n_negatives = self.__calc_dataset_balance(train_dataset)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_negatives / n_positives))

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
        log_file = config.get("log_file")
        
        losses = []
        _, best_eval_F1 = self.__mean_eval_loss(model, test_loader)

        iter_num = 0
        iter_time = time.time()
        max_iters = config.get("steps")

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
                mean_eval_loss, mean_eval_F1 = self.__mean_eval_loss(model, test_loader)

                tnow = time.time()
                iters_dt = tnow - iter_time
                iter_time = tnow

                if mean_eval_F1 > best_eval_F1:
                    continue
                best_eval_F1 = mean_eval_F1
                torch.save(model, config["checkpoint_file"])

                self.__print_and_log(
                    f"TRAIN - {iter_num=}, {mean_training_loss=}, {mean_eval_loss=}, {mean_eval_F1=}, {best_eval_F1=}, ({iters_dt:.2f}s)",
                    log_file,
                )

                losses = []
            
            # termination conditions
            if max_iters is not None and iter_num >= max_iters:
                break

    def __loss_step(self, batch, model, optimizer):
        loss = self.__calc_loss(model, batch)

        # backprop and update the parameters
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    @torch.no_grad()
    def __mean_eval_loss(self, model, data_loader):
        losses = []
        cmatrix = np.zeros((2, 2))
        model.eval()
        for batch in data_loader:
            loss, partial_cmatrix = self.__calc_loss(model, batch, True)
            losses.append(loss.item())
            cmatrix += partial_cmatrix
        model.train()
        F1 = self.__calc_f1_from_confusion_matrix(cmatrix)
        return np.mean(losses), F1

    def __calc_loss(self, model, batch, calc_confusion_matrix=False):
        input_ids = batch["input_ids"].to(self.config["device"])
        attention_mask = batch["attention_mask"].to(self.config["device"])
        labels = batch["labels"].to(self.config["device"])

        # get model output
        probs = model(input_ids, attention_mask)  # (BATCH_SIZE, 1)
        loss = self.loss_fn(probs, labels.float().view(-1, 1))
        if calc_confusion_matrix is False:
            return loss

        detached_probs = (probs.detach().cpu().numpy().reshape(-1) > 0.5)
        detached_labels = labels.detach().cpu().numpy().reshape(-1)
        cmatrix = confusion_matrix(detached_labels, detached_probs)

        return loss, cmatrix


    def __print_and_log(self, message, log_file):
        print(message)
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")

    def __calc_f1_from_confusion_matrix(self, cmatrix):
        _, FP, FN, TP = cmatrix.ravel()
    
        # Calculating Precision and Recall
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0

        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    def __calc_dataset_balance(self, dataset):
        cnts = [0, 0]
        for i in range(len(dataset)):
            cnts[dataset[i]["label"]] += 1
        return cnts
