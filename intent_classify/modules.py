import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .datasets import TextClassificationDataset
from .models import DistillBERTClassifier
from .utils import compute_accuracy


class IntentClassificationModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = DistillBERTClassifier(**config["model"])
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        xs, mask, ys = batch
        logits = self.model(xs, mask)
        loss = self.criterion(logits, ys)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, mask, ys = batch
        logits = self.model(xs, mask)
        loss = self.criterion(logits, ys)
        _, max_idx = torch.max(logits.data, dim=1)
        accuracy = compute_accuracy(max_idx, ys)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", accuracy, sync_dist=True)
        return loss

    def train_dataloader(self) -> DataLoader:
        train_ds = TextClassificationDataset(
            tokenizer=AutoTokenizer.from_pretrained(
                "distilbert/distilbert-base-uncased"
            ),
            **self.config["dataset"]["train"],
        )
        train_dl = DataLoader(
            dataset=train_ds,
            shuffle=True,
            **self.config["dataset"]["loaders"],
        )
        return train_dl

    def val_dataloader(self) -> DataLoader:
        val_ds = TextClassificationDataset(
            tokenizer=AutoTokenizer.from_pretrained(
                "distilbert/distilbert-base-uncased"
            ),
            **self.config["dataset"]["val"],
        )
        val_dl = DataLoader(
            dataset=val_ds,
            shuffle=False,
            **self.config["dataset"]["loaders"],
        )
        return val_dl

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), **self.config["optimizer"]
        )

        return {
            "optimizer": optimizer,
        }

    def export(self, filepath: str):
        checkpoint = {
            "state_dict": {
                "model": self.model.state_dict(),
            },
            "hyper_parameters": self.hparams,
        }
        torch.save(checkpoint, filepath)
        print(f'Model checkpoint has been saved to "{filepath}"')
