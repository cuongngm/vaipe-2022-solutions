import albumentations as A
from albumentations.pytorch import ToTensorV2
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
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def merge_data_train(df1, df2):
    df = []
    df.append(df1)
    df.append(df2)
    train_df = pd.concat(df, ignore_index=True)
    # skf = StratifiedKFold(n_splits=10)
    # for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.label)):
    #     train_df.loc[val_, "kfold"] = fold
    return train_df
    

class PillDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df
        self.transform = transform
        # self.transform = create_transform(
        #     input_size=(224, 224),
        #     crop_pct=1.0,
        # )
        self.image_paths = self.df["filepath"].values
        self.targets = self.df["label"].values
        
        
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[index]
        image_name = image_path.split('/')[-1]
        
        # image = Image.open(image_path).convert('RGB')
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            # image = self.transform(image)
            image = self.transform(image=img)['image']
        target = self.targets[index]
        target = torch.tensor(target, dtype=torch.long)

        return {"image_name": image_name, "image": image, "target": target}

    def __len__(self) -> int:
        return len(self.df)

    
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
        self.transform = {
            "train": A.Compose([
                A.Affine(rotate=(-15, 15), translate_percent=(0.0, 0.25), shear=(-3, 3), p=0.5),
                A.RandomResizedCrop(image_size, image_size, scale=(0.9, 1.0), ratio=(0.75, 1.333)), 
                A.HorizontalFlip(p=0.1), 
                A.Normalize( 
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.),

            "val": A.Compose([
                A.Resize(image_size, image_size), 
                A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.
                )
        }
        # self.transform = create_transform(
        #     input_size=(self.hparams.image_size, self.hparams.image_size),
        #     crop_pct=1.0,
        # )
        # self.train_transform = get_train_transforms(self.hparams.image_size)
        # self.valid_transform = get_valid_transforms(self.hparams.image_size)
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Split train df using fold
            # train_df = self.train_df[self.train_df.kfold != self.hparams.val_fold].reset_index(drop=True)
            # val_df = self.train_df[self.train_df.kfold == self.hparams.val_fold].reset_index(drop=True)
            self.train_dataset = PillDataset(self.train_df, transform=self.transform['train'])
            self.val_dataset = PillDataset(self.val_df, transform=self.transform['val'])

        if stage == "test" or stage is None:
            self.test_dataset = PillDataset(self.test_df, transform=self.transform['val'])

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