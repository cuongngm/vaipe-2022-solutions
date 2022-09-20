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

from dataset import LitDataModule, PillDataset, merge_data_train
from model import LitModule
from config import load_config
from eval.run import evaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from logult import setup_log


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
    logger = setup_log('saved', 'info.log')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    transform = {
            "train": A.Compose([
                A.Affine(rotate=(-15, 15), translate_percent=(0.0, 0.25), shear=(-3, 3), p=0.5),
                A.RandomResizedCrop(cfg.image_size, cfg.image_size, scale=(0.9, 1.0), ratio=(0.75, 1.333)), 
                A.HorizontalFlip(p=0.1), 
                A.Normalize( 
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.),

            "val": A.Compose([
                A.Resize(cfg.image_size, cfg.image_size), 
                A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.
                )
        }
 
    train_df = pd.read_csv('../data/crop/train_crop.csv')
    val_df = pd.read_csv('../data/crop/val_crop.csv')
    train_df = merge_data_train(train_df, val_df)
    train_dataset = PillDataset(train_df, transform['train'])
    train_dl = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_df = pd.read_csv('../data/crop/private_crop.csv')
    test_dataset = PillDataset(test_df, transform['val'])
    test_dl = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    print('load data sucessfull!!!')
    
    ENCODER_CLASSES_PATH = cfg.encoder_classes_path
    encoder = load_encoder()
    
    model_list = [# 'checkpoints/convnext_xlarge_384_in22ft1k_224-v1.ckpt',
                  '../checkpoints/convnext_large_384_in22ft1k_224.ckpt',
                  # 'checkpoints/convnext_large_384_in22ft1k_224-v1.ckpt',
                  '../checkpoints/convnext_xlarge_384_in22ft1k_224.ckpt',
                  # 'checkpoints_v3/tf_efficientnet_b7_ns_224.ckpt'
                 ]
    train_image_list = []
    train_embeddings_list = []
    train_targets_list = []
    test_image_list = []
    test_embeddings_list = []
    test_targets_list = []
    for idx in range(len(model_list)):
        weight_dir = model_list[idx]
        module = load_eval_module(weight_dir, device)
        train_image_names, train_embeddings, train_targets = get_embeddings(module, train_dl, encoder, stage="train")
        test_image_names, test_embeddings, test_targets = get_embeddings(module, test_dl, encoder, stage="test")
        # print('get size', train_embeddings.shape)
        # train_image_list.append(train_image_names)
        train_embeddings_list.append(train_embeddings)
        # train_targets_list.append(train_targets)
        # test_image_list.append(test_image_names)
        test_embeddings_list.append(test_embeddings)
        # test_targets_list.append(test_targets)
    train_embeddings = np.concatenate(train_embeddings_list, axis=1)
    test_embeddings = np.concatenate(test_embeddings_list, axis=1)
    
    np.save('saved/train_embeddings.npy', train_embeddings)
    np.save('saved/train_targets.npy', train_targets)
    # train_embeddings = np.load('saved/train_embed.npy')
    print('get size concat', train_embeddings.shape)
    # checkpoint_path = 'checkpoints/convnext_large_384_in22ft1k_224.ckpt'
    # module = load_eval_module(checkpoint_path, device)
    # print('load model sucessfull!!!')
    """
    train_dl, val_dl, test_dl = load_dataloaders(
            train_csv='../data/crop/train_crop.csv',
            val_csv='../data/crop/val_crop.csv',
            test_csv='../data/crop/test_crop.csv',
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            )
    
    print('load data sucessfull!!!')
    
    train_image_names, train_embeddings, train_targets = get_embeddings(module, train_dl, encoder, stage="train")
    val_image_names, val_embeddings, val_targets = get_embeddings(module, val_dl, encoder, stage="val")
    test_image_names, test_embeddings, test_targets = get_embeddings(module, test_dl, encoder, stage="test")
    print('get embedding sucessfull!!!')
    """
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
    df_detect = pd.read_csv('../data/detect_phase_private.csv')
    all_candidate = []
    list_img = df_detect['image_name'].tolist()
    for image_name in tqdm(list_img):
        list_candidate = []
        sub_df = distances_df[distances_df.image == image_name].reset_index(drop=True)
        u = dict()
        for idx2 in range(len(sub_df)):
            u[sub_df.loc[idx2, 'target']] = sub_df.loc[idx2, 'distances']
            list_candidate.append(u)
        list_candidate = [dict(t) for t in {tuple(d.items()) for d in list_candidate}]
        all_candidate.append(list_candidate)
        
    df_detect['candidate'] = all_candidate
    df_detect.to_csv('saved/merge.csv', index=False)
    drug_mapping = '../data/drug_private.npy'
    # drug_mapping = '../data/giang_mapping.npy'
    drug_mapping = np.load(drug_mapping, allow_pickle=True)
    drug_mapping = drug_mapping.tolist()
    # with open('../data/private_mapping.json', 'r') as fr:
    #     drug_mapping = json.load(fr)
    """
    # tuning threshold public val set
    threshold = [0.1*x for x in range(2, 8)]
    best_cv = 0
    for thresh in threshold:
        print('threshold', thresh)
        list_pred = []
        list_distance = []
        list_pill_in_res = []
        for idx in tqdm(range(len(df_detect))):
            candidate = df_detect.loc[idx, 'candidate']
            candidate = ast.literal_eval(str(candidate))
            candidate = sorted(candidate, key=lambda x:list(x.values())[0], reverse=True)

            pill_name = df_detect.loc[idx, 'image_name']
            pres_id = pill_name.split('_')[2]
            pres_name = 'VAIPE_P_TEST_{}'.format(pres_id)
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
                    if v < thresh:
                        continue
                    if int(k) in pill_id_in_pres:
                        check = 1
                        pred_id = k
                        distance = v
                        # if distance < 0.6:
                        #     distance = 0.8

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
        wmap, wmap50 = evaler('saved/results.csv', '../submission/gt_update.csv')
        print('wmap', wmap)
        print('wmap50', wmap50)
        if wmap > best_cv:
            best_thresh = thresh
            best_cv = wmap
        
    print('best threshold', best_thresh)
    print('best_cv', best_cv)
    """
    list_pred = []
    list_distance = []
    list_pill_in_res = []
    reverse = dict()
    with open('../data/private_test/pill_pres_map.json', 'r') as fr:
        data = json.load(fr)
        for k, v in data.items():
            for pill in v:
                if ' (1)' in pill:
                    continue
                reverse[pill[:-4]] = k
    # with open('../data/private_test/reverse.json', 'r') as fr:
    #     reverse = json.load(fr)
    for idx in tqdm(range(len(df_detect))):
        candidate = df_detect.loc[idx, 'candidate']
        pill_name = df_detect.loc[idx, 'image_name']
        pres_id = pill_name.split('_')[:-1]
        pres_name = reverse['_'.join(pill_name.split('_')[:-1])]
        # pres_id = pill_name.split('_')[2]
        # pres_name = 'VAIPE_P_TEST_{}'.format(pres_id)
        pill_id_in_pres = drug_mapping[pres_name]
        pill_id_in_pres = list(map(int, pill_id_in_pres))
        list_pill_in_res.append(pill_id_in_pres)
        if len(candidate) == 0:
            pred_id = 107
            list_pred.append(pred_id)
            list_distance.append(cfg.re_conf)
            distance = 0
            continue
        
        candidate = ast.literal_eval(str(candidate))
        candidate = dict(sorted(candidate[0].items(), key=lambda item: item[1], reverse=True))
        # candidate = sorted(candidate, key=lambda x:list(x.values())[0], reverse=True)
        # print('candidate', candidate)
        
        check = 0
        
        for k, v in candidate.items():
            if int(k) in pill_id_in_pres:
                check = 1
                pred_id = k
                distance = v
                if distance < 0.5:
                    distance = distance * 2
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
    df_detect.to_csv('saved/postprocess.csv', index=False)
    rs_df = pd.DataFrame()
    rs_df['image_name'] = df_detect['image_name'].apply(lambda x: '_'.join(x.split('_')[:-1]) + x[-4:])
    rs_df['class_id'] = df_detect['class_id'].apply(lambda x: int(x))
    rs_df['confidence_score'] = df_detect['distance']
    rs_df['x_min'] = df_detect['x_min']
    rs_df['y_min'] = df_detect['y_min']
    rs_df['x_max'] = df_detect['x_max']
    rs_df['y_max'] = df_detect['y_max']
    rs_df.to_csv('../submission/results.csv', index=False)
    print('End...')
    # wmap, wmap50 = evaler('submission/results.csv', '../submission/gt_update.csv')
    # print('wmap', wmap)
    # print('wmap50', wmap50)
