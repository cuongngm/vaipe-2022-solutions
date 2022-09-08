# tao tap tu dien thuoc theo don su dung model yolov5s va vietocr
import cv2
import numpy as np
import torch
import json
import os
import re
import distance
from PIL import Image
from tqdm import tqdm
import torch
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def xywh2yolo(x, y, w, h, w_img, h_img):
    # xmin, ymin, w, h
    x_mid = (x/w_img + x/w_img + w/w_img) / 2
    y_mid = (y/h_img + y/h_img + h/h_img) / 2
    w_norm = w / w_img
    h_norm = h / h_img
    return x_mid, y_mid, w_norm, h_norm


class PresOCR(object):
    def __init__(self):
        # self.config = Cfg.load_config_from_name("vgg_transformer")
        # self.config["weights"] = 'recognition/weights/new_transformer.pth'
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['weights'] = '../checkpoints/transformerocr.pth'
        self.config["device"] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config["cnn"]["pretrained"] = False
        self.config["predictor"]["beamsearch"] = False
        self.detector = Predictor(self.config)


    def recog(self, list_img):
        all_text = []
        for img in list_img:
            s = self.detector.predict(img)
            all_text.append(s)
        return all_text
    

def pres_ocr(all_pres_path, det_model, reg_model):
    drug = dict()
    for pres_path in all_pres_path:
        img = cv2.imread(pres_path)
        rs_pres = det_model(img, size=640, augment=False)
        preds = rs_pres.pandas().xyxy[0]
        bboxes = preds[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values
        # print(bboxes)
        all_img_pil = []
        for boxes in bboxes:
            boxes = boxes[:4]
            boxes = list(map(int, boxes.tolist()))
            img_crop = img[boxes[1]: boxes[3], boxes[0]: boxes[2]]
            img_pil = Image.fromarray(img_crop.astype('uint8'), 'RGB')
            all_img_pil.append(img_pil)
        text = reg_model.recog(all_img_pil)
        pres_name = pres_path.split('/')[-1][:-4]
        drug[pres_name] = text
        # print('name: {} === {}'.format(pres_name, text))
    return drug


def create_drugdict():
    reg_model = PresOCR()
    det_model = torch.hub.load('ultralytics/yolov5', 'custom', path='../checkpoints/pres.pt', force_reload=True)
    all_pres_path = []
    root = '../data/public_test_new/prescription/image'
    with open('../data/drug_dict.json', 'r') as fr:
        datas = json.load(fr)
    for filename in os.listdir(root):
        filepath = os.path.join(root, filename)
        all_pres_path.append(filepath)
    drugs = pres_ocr(all_pres_path, det_model, reg_model)
    pattern = '\d\)'
    cuong_mapping = dict()
    for k, v in tqdm(drugs.items()):
        list_candidate = []
        for drug in v:
            match = re.search(pattern, drug)
            if match is not None:
                drug = drug[match.span()[0]:]
            for data in datas:
                dis = distance.levenshtein(drug.lower(), data['drug_name'].lower())
                if dis < 4 and data['drug_id'] not in list_candidate:
                    print('checker pred: {} ====== gt: {}'.format(drug, data['drug_name']))
                    list_candidate.append(data['drug_id'])
        cuong_mapping[k] = list_candidate
    with open('../data/cuong_mapping.json', 'w') as fw:
        json.dump(cuong_mapping, fw)


def remove_noise_boxes(img, boxes):
    # remove box o ria cac anh
    h_img, w_img = img.shape[:2]
    xmin, ymin, xmax, ymax = boxes
    hbox = ymax - ymin
    wbox = xmax - xmin
    S = (hbox * wbox) / (h_img * w_img)
    if (xmin == 0 or ymin == 0 or xmax == w_img or ymax == h_img) and S< 0.02:
        return True
    else:
        return False
