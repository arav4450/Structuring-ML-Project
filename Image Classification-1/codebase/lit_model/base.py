"""Basic LightningModules on which other modules can be built."""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import StepLR


OPTIMIZER = "SGD"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100


class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        self.data_config = self.model.data_config
        self.mapping = self.data_config["mapping"]
        self.input_dims = self.data_config["input_dims"]

        # optimizer
        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        # learning rate
        self.lr = self.args.get("lr", LR)
        # loss function
        loss = self.args.get("loss", LOSS)
        self.loss_fn = getattr(torch.nn.functional, loss)
        # learning rate scheduling
        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)
        # evaluation metric
        self.train_acc = Accuracy(task = 'binary')
        self.val_acc = Accuracy(task = 'binary')
        self.test_acc = Accuracy(task = 'binary')

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser
    
    @classmethod
    def log_dirname(cls):
        return Path(__file__).resolve().parents[2] / "logs"

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = StepLR(
            optimizer=optimizer, step_size = 7, gamma=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation/loss"}

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        logits = self.model(x)
        return torch.argmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        predictions = torch.nn.functional.softmax(logits, dim=1)
        self.train_acc(torch.argmax(predictions, dim=1), y)

        self.log("train/loss", loss, on_step=False, on_epoch=True,logger=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,logger=True)

        outputs = {"loss": loss}

        return outputs

    def _run_on_batch(self, batch, with_preds=False):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        return x, y, logits, loss

    def validation_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        predictions = torch.nn.functional.softmax(logits, dim=1)
        self.val_acc(torch.argmax(predictions, dim=1),y)

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True,logger=True)
        self.log("validation/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True,logger=True)

        outputs = {"loss": loss}

        return outputs

    def test_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        predictions = torch.nn.functional.softmax(logits, dim=1)
        self.test_acc(torch.argmax(predictions , dim=1), y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)