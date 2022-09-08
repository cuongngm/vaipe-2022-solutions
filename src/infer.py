import numpy as np
import ast
import json
import torch
import pytorch_lightning as pl
import pandas as pd
from tqdm import tqdm
from typing import Callable, Dict, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import NearestNeighbors

from dataset import LitDataModule
from model import LitModule
from config import load_config



# inference
def load_eval_module(checkpoint_path: str, device: torch.device) -> LitModule:
    module = LitModule.load_from_checkpoint(checkpoint_path)
    module.to(device)
    module.eval()

    return module


def load_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    datamodule = LitDataModule(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    datamodule.setup()

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    return train_dl, val_dl, test_dl


def load_encoder() -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.classes_ = np.load(ENCODER_CLASSES_PATH, allow_pickle=True)

    return encoder


@torch.inference_mode()
def get_embeddings(
    module: pl.LightningModule, dataloader: DataLoader, encoder: LabelEncoder, stage: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_image_names = []
    all_embeddings = []
    all_targets = []

    for batch in tqdm(dataloader, desc=f"Creating {stage} embeddings"):
        image_names = batch["image_name"]
        images = batch["image"].to(module.device)
        targets = batch["target"].to(module.device)

        embeddings = module(images)

        all_image_names.append(image_names)
        all_embeddings.append(embeddings.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_image_names = np.concatenate(all_image_names)
    all_embeddings = np.vstack(all_embeddings)
    all_targets = np.concatenate(all_targets)

    all_embeddings = normalize(all_embeddings, axis=1, norm="l2")
    all_targets = encoder.inverse_transform(all_targets)

    return all_image_names, all_embeddings, all_targets


if __name__ == '__main__':
    cfg = load_config('../config/default.yaml')
    train_df = pd.read_csv('../data/crop/train_crop.csv')
    encoder = LabelEncoder()
    train_df['label'] = encoder.fit_transform(train_df['label'])
    np.save('../data/encoder_classes.npy', encoder.classes_)
    checkpoint_path = '../checkpoints/convnext_large_384_in22ft1k_224.ckpt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    module = load_eval_module(checkpoint_path, device)
    print('load model sucessfull!!!')
    train_dl, val_dl, test_dl = load_dataloaders(
            train_csv='../data/crop/train_crop.csv',
            val_csv='../data/crop/val_crop.csv',
            test_csv='../data/crop/test_crop.csv',
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            )
    print('load data sucessfull!!!')
    ENCODER_CLASSES_PATH = cfg.encoder_classes_path
    encoder = load_encoder()
    train_image_names, train_embeddings, train_targets = get_embeddings(module, train_dl, encoder, stage="train")
    val_image_names, val_embeddings, val_targets = get_embeddings(module, val_dl, encoder, stage="val")
    test_image_names, test_embeddings, test_targets = get_embeddings(module, test_dl, encoder, stage="test")
    print('get embedding sucessfull!!!')
    # train_embeddings = np.concatenate([train_embeddings, val_embeddings])
    # train_targets = np.concatenate([train_targets, val_targets])
    neigh = NearestNeighbors(n_neighbors=cfg.n_neightbors, metric="cosine")
    neigh.fit(train_embeddings)
    test_dist, test_cosine_idx = neigh.kneighbors(test_embeddings, return_distance=True)
    test_cosine = 1 - test_dist
    distances_df = []
    for i, image_name in tqdm(enumerate(test_image_names), desc=f"Creating test_df"):
        target = train_targets[test_cosine_idx[i]]
        distances = test_cosine[i]
        subset_preds = pd.DataFrame(np.stack([target, distances], axis=1), columns=["target", "distances"])
        subset_preds["image"] = image_name
        distances_df.append(subset_preds)

    distances_df = pd.concat(distances_df).reset_index(drop=True)

    # distances_df = distances_df.groupby(["image", "target"]).distances.nlargest(3).reset_index()
    distances_df = distances_df.groupby(["image", "target"]).distances.max().reset_index()
    distances_df.drop(distances_df[distances_df.distances < cfg.threshold].index, inplace=True)
    distances_df = distances_df.reset_index(drop=True)
    print(distances_df.head())
    distances_df.to_csv('saved/arcface.csv', index=False)
    df_detect = pd.read_csv('../data/detect_phase.csv')
    all_candidate = []
    list_img = df_detect['image_name'].tolist()
    for image_name in tqdm(list_img):
        # print(image_name)
        list_candidate = []
        check_bef = 0
        check_af = 0
        for idx2 in range(len(distances_df)):
            if distances_df.loc[idx2, 'image'] == image_name:
                check_bef = 1
                u = dict()
                u[distances_df.loc[idx2, 'target']] = distances_df.loc[idx2, 'distances']
                list_candidate.append(u)
            else:
                if check_bef == 1:
                    check_af = 1
                else:
                    check_af = 0
            if check_af == 1:
                break
        all_candidate.append(list_candidate)
    df_detect['candidate'] = all_candidate
    df_detect.to_csv('saved/merge.csv', index=False)
    # drug_mapping = '../data/giang/drug_new.npy'
    # drug_mapping = np.load(drug_mapping, allow_pickle=True)
    # drug_mapping = drug_mapping.tolist()
    with open('../data/cuong_mapping.json', 'r') as fr:
        drug_mapping = json.load(fr)

    list_pred = []
    list_distance = []
    list_pill_in_res = []
    for idx in tqdm(range(len(df_detect))):
        candidate = df_detect.loc[idx, 'candidate']
        candidate = ast.literal_eval(str(candidate))
        candidate = sorted(candidate, key=lambda x:list(x.values())[0], reverse=True)
        
        pill_name = df_detect.loc[idx, 'image_name']
        pres_id = pill_name.split('_')[2]
        pres_name = 'VAIPE_P_TEST_NEW_{}'.format(pres_id)
        pill_id_in_pres = drug_mapping[pres_name]
        pill_id_in_pres = list(map(int, pill_id_in_pres))
        list_pill_in_res.append(pill_id_in_pres)
        check = 0
        if len(candidate) == 0:
            pred_id = 107
            list_pred.append(pred_id)
            list_distance.append(cfg.re_conf)
            distance = 0
            continue
        for can in candidate:
            for k, v in can.items():
                if int(k) in pill_id_in_pres:
                    check = 1
                    pred_id = k
                    distance = v

            if check == 1:
                break
        if check == 0:
            pred_id = 107
            distance = cfg.re_conf
        list_pred.append(pred_id)
        list_distance.append(distance)
        
    df_detect['class_id'] = list_pred
    df_detect['distance'] = list_distance
    df_detect['list_pill_in_res'] = list_pill_in_res
    
    rs_df = pd.DataFrame()
    rs_df['image_name'] = df_detect['image_name'].apply(lambda x: '_'.join(x.split('_')[:-1]) + '.jpg')
    rs_df['class_id'] = df_detect['class_id'].apply(lambda x: int(x))
    rs_df['confidence_score'] = df_detect['distance']
    rs_df['x_min'] = df_detect['x_min']
    rs_df['y_min'] = df_detect['y_min']
    rs_df['x_max'] = df_detect['x_max']
    rs_df['y_max'] = df_detect['y_max']
    rs_df.to_csv('saved/results.csv', index=False)
    print('End...')
    # end...
