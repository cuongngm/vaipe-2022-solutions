import torch
import pytorch_lightning as pl
from dataset import LitDataModule
from model import LitModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from config import load_config


def train(
    cfg
):
    pl.seed_everything(42)

    datamodule = LitDataModule(
        train_csv=cfg.train_csv,
        val_csv=cfg.val_csv,
        test_csv=cfg.test_csv,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    
    datamodule.setup()
    len_train_dl = len(datamodule.train_dataloader())

    module = LitModule(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        drop_rate=cfg.drop_rate,
        embedding_size=cfg.embedding_size,
        num_classes=cfg.num_classes,
        arc_s=cfg.arc_s,
        arc_m=cfg.arc_m,
        arc_easy_margin=cfg.arc_easy_margin,
        arc_ls_eps=cfg.arc_ls_eps,
        optimizer=cfg.optimizer,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        len_train_dl=len_train_dl,
        epochs=cfg.max_epochs
    )
    
    model_checkpoint = ModelCheckpoint(
        cfg.checkpoints_dir,
        filename=f"{cfg.model_name}_{cfg.image_size}",
        monitor="val_loss",
    )
        
    trainer = pl.Trainer(
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        auto_lr_find=cfg.auto_lr_find,
        auto_scale_batch_size=cfg.auto_scale_batch_size,
        benchmark=True,
        callbacks=[model_checkpoint],
        deterministic=True,
        fast_dev_run=cfg.fast_dev_run,
        gpus=cfg.gpus,
        max_epochs=2 if cfg.debug else cfg.max_epochs,
        precision=cfg.precision,
        stochastic_weight_avg=cfg.stochastic_weight_avg,
        limit_train_batches=0.1 if cfg.debug else 1.0,
        limit_val_batches=0.1 if cfg.debug else 1.0,
    )

    trainer.tune(module, datamodule=datamodule)

    trainer.fit(module, datamodule=datamodule)
    

if __name__ == '__main__':
    cfg = load_config('../config/default.yaml')
    train(cfg)
