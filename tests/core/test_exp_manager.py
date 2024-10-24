# coding=utf-8
__author__ = "Dimitris Karkalousos"

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from atommic.core.classes.modelPT import ModelPT


class MyTestOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        self._step = 0
        super().__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if self._step == 0:
                    p.data = 0.1 * torch.ones(p.shape)
                elif self._step == 1:
                    p.data = 0.0 * torch.ones(p.shape)
                else:
                    p.data = 0.01 * torch.ones(p.shape)
        self._step += 1
        return loss


class DoNothingOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        self._step = 0
        super().__init__(params, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step += 1
        return loss


class OnesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_len):
        super().__init__()
        self.__dataset_len = dataset_len

    def __getitem__(self, *args):
        return torch.ones(2)

    def __len__(self):
        return self.__dataset_len


class ExampleModel(ModelPT):
    def __init__(self, *args, **kwargs):
        cfg = OmegaConf.structured({})

        trainer = OmegaConf.create(
            {
                "strategy": "ddp",
                "accelerator": "cpu",
                "num_nodes": 1,
                "max_epochs": 20,
                "precision": 32,
                "enable_checkpointing": False,
                "logger": False,
                "log_every_n_steps": 50,
                "check_val_every_n_epoch": -1,
                "max_steps": -1,
            }
        )
        trainer = OmegaConf.create(OmegaConf.to_container(trainer, resolve=True))
        trainer = pl.Trainer(**trainer)

        super().__init__(cfg=cfg, trainer=trainer)
        pl.seed_everything(1234)
        self.l1 = torch.nn.modules.Linear(in_features=2, out_features=1)

    @staticmethod
    def train_dataloader():
        dataset = OnesDataset(2)
        return torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8)

    @staticmethod
    def val_dataloader():
        dataset = OnesDataset(10)
        return torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8)

    def forward(self, batch):
        output = self.l1(batch)
        output = torch.nn.functional.l1_loss(output, torch.zeros(output.size()).to(output.device))
        return output

    def validation_step(self, batch, batch_idx):
        self.loss = self(batch)
        return self.loss

    def training_step(self, batch, batch_idx):
        return self(batch)

    def configure_optimizers(self):
        return MyTestOptimizer(self.parameters())
        # return torch.optim.Adam(self.parameters(), lr=0.1)

    def list_available_models(self):
        raise NotImplementedError()

    def setup_training_data(self):
        raise NotImplementedError()

    def setup_validation_data(self):
        raise NotImplementedError()

    def on_validation_epoch_end(self):
        self.log("val_loss", torch.stack([self.loss]).mean())


class DoNothingModel(ExampleModel):
    def configure_optimizers(self):
        return DoNothingOptimizer(self.parameters())
