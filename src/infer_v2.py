import torch
import json
import os
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from typing import *
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import NearestNeighbors
from logult import setup_log

from model import LitModule
from dataset import PillDataset, PillInferDataset
from config import load_config
from util import remove_noise_boxes


# inference
def load_eval_module(checkpoint_path: str, device: torch.device) -> LitModule:
    module = LitModule.load_from_checkpoint(checkpoint_path)
    module.to(device)
    module.eval()
    return module


def load_encoder() -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.classes_ = np.load(ENCODER_CLASSES_PATH, allow_pickle=True)

    return encoder


def load_dataloaders(
    train_df,
    list_img: list,
    list_img_name: list,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:

    datamodule = Lit2(
        train_df=train_df,
        list_img=list_img,
        list_img_name=list_img_name,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    datamodule.setup()

    train_dl = datamodule.train_dataloader()
    test_dl = datamodule.test_dataloader()

    return train_dl, test_dl


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
    # logger
    logger = setup_log('saved', 'info.log')
    logger.info('VAIPE public test inference')
    logger.info('........................')
    # cfg
    cfg = load_config('../config/default.yaml')
    logger.info('Load config file: \n {}'.format(cfg))
    # model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detect_path = 'checkpoints/pill_2.pt'
    recog_path = 'checkpoints/convnext_large_384_in22ft1k_224.ckpt'
    detect_pill_model = torch.hub.load('ultralytics/yolov5', 'custom', path=detect_path, force_reload=True)
    recog_pill_model = load_eval_module(recog_path, device)
    logger.info('Load model sucessul!!')
    
    ENCODER_CLASSES_PATH = '../data/kaggle/encoder_classes.npy'
    encoder = load_encoder()
    
    # data
    df_train = pd.read_csv('../data/crop/train_crop.csv')
    df_val = pd.read_csv('../data/crop/val_crop.csv')
    pres_image_test = '../data/public_test/prescription/image'
    pill_image_test = '../data/public_test/pill/image'
    
    # drug_mapping = '../data/giang_mapping.npy'
    # drug_mapping = np.load(drug_mapping, allow_pickle=True)
    # drug_mapping = drug_mapping.tolist()
    with open('../data/cuong_mapping.json', 'r') as fr:
        drug_mapping = json.load(fr)

    # inference    
    pill_pres_map = '../data/public_test/pill_pres_map.json'
    results = []
    with open(pill_pres_map, 'r') as fr:
        datas = json.load(fr)
    for data in datas:
        pres_json_file = data['pres']
        pres_img_name = pres_json_file.replace('.json', '')
        pill_id_in_pres = drug_mapping[pres_img_name]
        pill_id_in_pres = list(map(int, pill_id_in_pres))
        logger.info('Pill id in prescription: {}'.format(pill_id_in_pres))
        frames = []
        for pres in pill_id_in_pres:
            frames.append(df_train[df_train['label'] == pres])
            frames.append(df_val[df_val['label'] == pres])
        sub_df = pd.concat(frames)
        train_dataset = PillDataset(sub_df)
        train_dl = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        train_image_names, train_embeddings, train_targets = get_embeddings(recog_pill_model, train_dl, encoder, stage="train")
        
        list_pill_json_file = data['pill']
        for pill_json_file in list_pill_json_file:
            list_pill = []
            list_pill_name = []
            list_boxes = []
          
            pill_img_name = pill_json_file + '.jpg'
            pill_img_path = os.path.join(pill_image_test, pill_img_name)
            
            img = cv2.imread(pill_img_path)
            rs_pill = detect_pill_model(img, size=1280, augment=False)
            preds = rs_pill.pandas().xyxy[0]
            bboxes = preds[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values
            list_conf = []
            list_pred_pill = []
            for idx, boxes in enumerate(bboxes):
                conf = boxes[4]
                if conf < 0.45:
                    continue
                if remove_noise_boxes(img, boxes[:4]):
                    continue
                boxes = boxes[:4]
                boxes = list(map(int, boxes.tolist()))
                crop = img[boxes[1]: boxes[3], boxes[0]: boxes[2]]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                list_pill.append(Image.fromarray(crop))
                list_pill_name.append(pill_json_file + '_' + str(idx))
                list_boxes.append(boxes)
            # create val df, test df
            print('num of pill', len(list_pill_name))
            test_dataset = PillInferDataset(list_pill, list_pill_name)
            test_dl = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
            test_image_names, test_embeddings, test_targets = get_embeddings(recog_pill_model, test_dl, encoder, stage="test")
            print('load data suceesfull!!!')
       
            neigh = NearestNeighbors(n_neighbors=len(sub_df) - 4, metric="cosine")
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
            distances_df = distances_df.groupby(["image", "target"]).distances.nlargest(3).reset_index()
            distances_df = distances_df.groupby(["image", "target"]).distances.mean().reset_index()
            # distances_df.drop(distances_df[distances_df.distances < 0.6].index, inplace=True)
            for idx, (pill_name, box) in enumerate(zip(list_pill_name, list_boxes)):
                infos = distances_df[distances_df['image'] == pill_name]
                infos = infos.sort_values('distances', ascending=False).reset_index()
                print(infos.head())
                logger.info(infos.loc[0, 'image'])
                check = dict()
                for idx in range(len(infos)):
                    logger.info('Predict id = {}, confidence = {}'.format(infos.loc[idx, 'target'], infos.loc[idx, 'distances']))
                    check[infos.loc[idx, 'target']] = infos.loc[idx, 'distances']
                logger.info('===================')
                image_name = infos.loc[0, 'image']
                image_name = '_'.join(image_name.split('_')[:-1]) + '.jpg'
                confidence = infos.loc[0, 'distances']
                
                xmin, ymin, xmax, ymax = box
                if confidence < 0.5:
                    # pred_id = 107
                    # confidence = 1
                    results.append([image_name, 107, 0.8, xmin, ymin, xmax, ymax, check])
                else:
                    pred_id = infos.loc[0, 'target']
                    results.append([image_name, pred_id, confidence, xmin, ymin, xmax, ymax, check])
            
    df = pd.DataFrame(results, columns=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max', 'check'])
    # image_name, class_id, confidence_score, xmin, ymin, xmax, ymax
    df.to_csv('results.csv', index=False)
    logger.info('Complete!!!')
