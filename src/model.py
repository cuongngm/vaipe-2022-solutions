import math
import timm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Callable, Dict, Optional, Tuple
from timm.optim import create_optimizer_v2


class WarmupCosineLambda:
    def __init__(self, warmup_steps: int, cycle_steps: int, decay_scale: float, exponential_warmup: bool = False):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.decay_scale = decay_scale
        self.exponential_warmup = exponential_warmup

    def __call__(self, epoch: int):
        if epoch < self.warmup_steps:
            if self.exponential_warmup:
                return self.decay_scale * pow(self.decay_scale, -epoch / self.warmup_steps)
            ratio = epoch / self.warmup_steps
        else:
            ratio = (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.cycle_steps)) / 2
        return self.decay_scale + (1 - self.decay_scale) * ratio
    
    
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float,
        m: float,
        easy_margin: bool,
        ls_eps: float,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
    
    

class LitModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_rate: float,
        embedding_size: int,
        num_classes: int,
        arc_s: float,
        arc_m: float,
        arc_easy_margin: bool,
        arc_ls_eps: float,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        len_train_dl: int,
        epochs:int
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = timm.create_model(model_name, pretrained=pretrained, drop_rate=drop_rate)
        self.embedding = nn.Linear(self.model.get_classifier().in_features, embedding_size)
        self.model.reset_classifier(num_classes=0, global_pool="avg")
        # in_features = self.model.get_classifier().in_features
        # print('in features', in_features)
        # self.pooling = GeM()
        # self.embedding = nn.Sequential(
        #                     nn.BatchNorm1d(in_features),
        #                     nn.Linear(in_features, embedding_size)
        #                     )
        self.arc = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=arc_s,
            m=arc_m,
            easy_margin=arc_easy_margin,
            ls_eps=arc_ls_eps,
        )

        self.loss_fn = F.cross_entropy

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        # pooled_features = self.pooling(features).flatten(1)
        # embeddings = self.embedding(pooled_features)
        embeddings = self.embedding(features)

        return embeddings

    def configure_optimizers(self):
        """
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt=self.hparams.optimizer,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            steps_per_epoch=self.hparams.len_train_dl,
            epochs=self.hparams.epochs,
        )
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1.0e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=1e-7)
        # warmup_steps = epochs * 0.2
        # cycle_steps = epochs - warmup_steps
        # lr_lambda = WarmupCosineLambda(warmup_steps, cycle_steps, 1.0e-2)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        images, targets = batch["image"], batch["target"]

        embeddings = self(images)
        outputs = self.arc(embeddings, targets, self.device)

        loss = self.loss_fn(outputs, targets)
        
        self.log(f"{step}_loss", loss)

        return loss
