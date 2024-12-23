import torch
from torch import nn, optim
import pytorch_lightning as pl
from typing import Optional
from model.sat import GraphTransformer


class SBMWrapper(pl.LightningModule):
    def __init__(
        self,
        model: GraphTransformer,
        num_class: int,
        abs_pe: Optional[str],
        learning_rate: float,
        weight_decay: float,
        lr_scheduler: nn.Module,
    ):
        super().__init__()
        self.model = model
        self.num_class = num_class
        self.abs_pe = abs_pe
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler

        self.train_loss = 0.0
        self.val_loss = 0.0
        self.test_loss = 0.0
        self.train_samples = 0
        self.val_samples = 0
        self.test_samples = 0

        self.val_acc = 0.0
        self.test_acc = 0.0

        self.save_hyperparameters(ignore=["model", "criterion"])

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
        loss = self.criterion(output, batch.y.squeeze())
        size = len(batch.y)

        self.train_loss += loss.item() * size
        self.train_samples += size
        
        correct = torch.sum(torch.argmax(output, dim=-1) == batch.y.squeeze())
        self.log("train/acc", correct / len(batch.y))

        return loss

    def on_train_epoch_end(self):
        train_loss = self.train_loss / self.train_samples
        self.log("train/loss", train_loss, prog_bar=True)

        self.train_loss = 0.0
        self.train_samples = 0

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.criterion(output, batch.y.squeeze())
        size = len(batch.y)

        self.val_loss += loss.item() * size
        self.val_samples += size

        correct = torch.sum(torch.argmax(output, dim=-1) == batch.y.squeeze())
        self.log("val/acc", correct / len(batch.y))

    def on_validation_epoch_end(self):
        val_loss = self.val_loss / self.val_samples
        self.log("val/loss", val_loss, prog_bar=True)

        self.val_loss = 0.0
        self.val_samples = 0

    def test_step(self, batch, batch_idx):
        output = self(batch)
        loss = self.criterion(output, batch.y.squeeze())
        size = len(batch.y)

        self.test_loss += loss.item() * size
        self.test_samples += size

        correct = torch.sum(torch.argmax(output, dim=-1) == batch.y.squeeze())
        self.log("test/acc", correct / len(batch.y))

    def on_test_epoch_end(self):
        test_loss = self.test_loss / self.test_samples
        self.log("test/loss", test_loss, prog_bar=True)

        self.test_loss = 0.0
        self.test_samples = 0

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = (
            {"scheduler": self.lr_scheduler(optimizer), "interval": "step"}
            if self.lr_scheduler
            else None
        )
        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer

    def criterion(self, y_hat, y):
        V = y.size(0)
        label_count = torch.bincount(y)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.num_class).long()
        cluster_sizes = cluster_sizes.to(self.device)
        cluster_sizes[torch.unique(y)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        ce = nn.CrossEntropyLoss(weight=weight)
        return ce(y_hat, y)