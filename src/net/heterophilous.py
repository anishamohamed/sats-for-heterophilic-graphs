import torch
from torch import nn, optim
import pytorch_lightning as pl
from typing import Optional
from model.sat import GraphTransformer


class HeterophilousGraphWrapper(pl.LightningModule):
    def __init__(
        self,
        model: GraphTransformer,
        abs_pe: Optional[str],
        learning_rate: float,
        weight_decay: float,
        lr_scheduler: nn.Module,
        mask: int,
    ):
        super().__init__()

        self.model = model
        self.abs_pe = abs_pe
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.mask = mask

        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=["model", "criterion"])

    def forward(self, data):
        # data: (num_nodes, num_features)
        return self.model(data)

    def training_step(self, data, data_idx):
        if self.abs_pe == "lap":
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(data.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.abs_pe = data.abs_pe * sign_flip.unsqueeze(0)

        output = self(data)[data.train_mask[:, self.mask]]
        y = data.y[data.train_mask[:, self.mask]].squeeze()

        loss = self.criterion(output, y)
        self.log("train/loss", loss.item(), prog_bar=True)
        correct = torch.sum(torch.argmax(output, dim=-1) == y)
        self.log("train/acc", correct / len(y))

        return loss

    def validation_step(self, data, data_idx):
        output = self(data)[data.val_mask[:, self.mask]]
        y = data.y[data.val_mask[:, self.mask]].squeeze()

        loss = self.criterion(output, y)
        self.log("val/loss", loss.item(), prog_bar=True)
        predictions = torch.argmax(output, dim=-1)
        correct = torch.sum(torch.argmax(output, dim=-1) == y)
        self.log("val/acc", correct / len(y))

    def test_step(self, data, data_idx):
        output = self(data)[data.test_mask[:, self.mask]]
        y = data.y[data.test_mask[:, self.mask]].squeeze()

        loss = self.criterion(output, y)
        self.log("test/loss", loss.item(), prog_bar=True)
        correct = torch.sum(torch.argmax(output, dim=-1) == y)
        self.log("test/acc", correct / len(y))

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
