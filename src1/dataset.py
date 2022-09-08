import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from timm.data.transforms_factory import create_transform
from typing import Optional, Callable, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from config import load_config


class PillDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df
        # self.transform = transform
        self.transform = create_transform(
            input_size=(224, 224),
            crop_pct=1.0,
        )
        self.image_paths = self.df["filepath"].values
        self.targets = self.df["label"].values

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[index]
        image_name = image_path.split('/')[-1]
        
        image = Image.open(image_path).convert('RGB')
        # img = get_img(image_path)
        if self.transform:
            image = self.transform(image)
            # image = self.transform(image=img)['image']
        target = self.targets[index]
        target = torch.tensor(target, dtype=torch.long)

        return {"image_name": image_name, "image": image, "target": target}

    def __len__(self) -> int:
        return len(self.df)


class PillInferDataset(Dataset):
    def __init__(self, list_img, list_img_name, transform: Optional[Callable] = None):
        self.list_img = list_img
        self.list_img_name = list_img_name
        self.transform = self.transform = create_transform(
            input_size=(224, 224),
            crop_pct=1.0,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image = self.list_img[index]
        if self.transform:
            image = self.transform(image)
            # image = self.transform(image=img)['image']
        target = 0 
        target = torch.tensor(target, dtype=torch.long)
        image_name = self.list_img_name[index]  
        return {"image_name": image_name, "image": image, "target": target}

    def __len__(self) -> int:
        return len(self.list_img)
    

class Lit2(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        list_img: list,
        list_img_name: list,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()
        
        self.train_df = train_df
        self.list_img = list_img
        self.list_img_name = list_img_name
        self.transform = create_transform(
            input_size=(self.hparams.image_size, self.hparams.image_size),
            crop_pct=1.0,
        )
        
        # self.train_transform = get_train_transforms(self.hparams.image_size)
        # self.valid_transform = get_valid_transforms(self.hparams.image_size)
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Split train df using fold
            # train_df = self.train_df[self.train_df.kfold != self.hparams.val_fold].reset_index(drop=True)
            # val_df = self.train_df[self.train_df.kfold == self.hparams.val_fold].reset_index(drop=True)
            self.train_dataset = PillDataset(self.train_df, transform=self.transform)
       
        if stage == "test" or stage is None:
            self.test_dataset = PillInferDataset(self.list_img, self.list_img_name, self.transform)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: PillDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=train,
        )
    
    
class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        test_csv: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv)
        self.val_df = pd.read_csv(val_csv)
        self.test_df = pd.read_csv(test_csv)
        
        self.transform = create_transform(
            input_size=(self.hparams.image_size, self.hparams.image_size),
            crop_pct=1.0,
        )
        # self.train_transform = get_train_transforms(self.hparams.image_size)
        # self.valid_transform = get_valid_transforms(self.hparams.image_size)
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Split train df using fold
            # train_df = self.train_df[self.train_df.kfold != self.hparams.val_fold].reset_index(drop=True)
            # val_df = self.train_df[self.train_df.kfold == self.hparams.val_fold].reset_index(drop=True)
            self.train_dataset = PillDataset(self.train_df, transform=self.transform)
            self.val_dataset = PillDataset(self.val_df, transform=self.transform)

        if stage == "test" or stage is None:
            self.test_dataset = PillDataset(self.test_df, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: PillDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=train,
        )
