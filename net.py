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
        transductive: bool = False,   
    ):
        super().__init__()
        self.model = model
        self.abs_pe = abs_pe
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.transductive = transductive

        self.train_loss = 0.0
        self.val_loss = 0.0
        self.test_loss = 0.0
        self.train_samples = 0
        self.val_samples = 0
        self.test_samples = 0

        self.save_hyperparameters(ignore=['model', 'criterion'])

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        if self.abs_pe == "lap":
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(batch.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch.abs_pe = batch.abs_pe * sign_flip.unsqueeze(0)
        output = self(batch)

        if self.transductive:
            output = output[batch.train_mask]
            y = batch.y[batch.train_mask]
        else:
            y = batch.y

        loss = self.criterion(output, y)
        size = len(y)

        self.train_loss += loss.item() * size
        self.train_samples += size

        return loss
    
    def on_train_epoch_end(self):
        train_loss = self.train_loss / self.train_samples
        self.log("train/loss", train_loss, prog_bar=True)

        self.train_loss = 0.0
        self.train_samples = 0

    def validation_step(self, batch, batch_idx):
        output = self(batch)

        if self.transductive:
            output = output[batch.val_mask]
            y = batch.y[batch.val_mask]
        else:
            y = batch.y
            
        loss = self.criterion(output, y)
        size = len(y)
        
        self.val_loss += loss.item() * size
        self.val_samples += size

    def on_validation_epoch_end(self):
        val_loss = self.val_loss / self.val_samples
        self.log("val/loss", val_loss, prog_bar=True)

        self.val_loss = 0.0
        self.val_samples = 0

    def test_step(self, batch, batch_idx):
        output = self(batch)

        if self.transductive:
            output = output[batch.test_mask]
            y = batch.y[batch.test_mask]
        else:
            y = batch.y
            
        loss = self.criterion(output, y)
        size = len(y)

        self.test_loss += loss.item() * size
        self.test_samples += size

    def on_test_epoch_end(self):
        test_loss = self.test_loss / self.test_samples
        self.log("test/loss", test_loss, prog_bar=True)

        self.test_loss = 0.0
        self.test_samples = 0
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = {'scheduler': self.lr_scheduler(optimizer), 'interval': 'step'} if self.lr_scheduler else None
        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer
