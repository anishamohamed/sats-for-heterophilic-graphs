import torch
import pytorch_lightning as pl
from torch import nn, optim
import numpy as np

from typing import Optional

from model.sat import GraphTransformer

class GraphTransformerWrapper(pl.LightningModule):
    def __init__(
        self,
        model: GraphTransformer,
        abs_pe: Optional[str],
        criterion: nn.Module,
        learning_rate: float,
        weight_decay: float,
        lr_scheduler: nn.Module,    
    ):
        super().__init__()
        self.model = model
        self.abs_pe = abs_pe
        self.criterion = criterion
        # self.val_metric = val_metric
        # self.collect_val_metric = collect_val_metric
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        # self.val_metric = []
        # self.test_metric = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.abs_pe == "lap":
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(batch.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch.abs_pe = batch.abs_pe * sign_flip.unsqueeze(0)
        output = self(batch)
        loss = self.criterion(output, batch.y)
        self.train_loss.append(loss.detach().cpu().numpy())

        return loss
    
    def on_train_epoch_end(self):
        train_loss = np.mean(self.train_loss)
        self.log("train/loss", train_loss, prog_bar=True)

        self.train_loss.clear()

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.criterion(output, batch.y)
        # metric = self.val_metric(output, batch.y)
        self.val_loss.append(loss.detach().cpu().numpy())
        # self.val_metric.append(metric.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        val_loss = np.mean(self.val_loss)
        # val_metric = np.mean(self.val_metric)
        self.log("val/loss", val_loss, prog_bar=True)
        # self.log("val/metric", val_metric, prog_bar=True)

        self.val_loss.clear()
        # self.val_metric.clear()

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.criterion(output, batch.y)
        # metric = self.val_metric(output, batch.y)
        self.test_loss.append(loss.detach().cpu().numpy())
        # self.test_metric.append(metric.detach().cpu().numpy())

    def on_test_epoch_end(self):
        test_loss = np.mean(self.test_loss)
        # test_metric = self.collect_val_metric(self.test_metric)
        self.log("test/loss", test_loss, prog_bar=True)
        # self.log("test/metric", test_metric, prog_bar=True)

        self.test_loss.clear()
        # self.test_metric.clear()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = {'scheduler': self.lr_scheduler(optimizer), 'interval': 'step'} if self.lr_scheduler else None
        return [optimizer], [scheduler]
