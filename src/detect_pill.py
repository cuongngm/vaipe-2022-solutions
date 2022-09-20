import os
import torch
import json
import cv2
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from logult import setup_log
from util import remove_noise_boxes


def detect(pill_img_path):
    name = pill_img_path.split('/')[-1]
    print(name)
    list_rs = []
    img = cv2.imread(pill_img_path)
    rs_pill = detect_pill_model(img, size=1280, augment=False)
    preds = rs_pill.pandas().xyxy[0]
    bboxes = preds[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values
    list_conf = []
    list_pred_pill = []
    for idx, boxes in enumerate(bboxes):
        conf = boxes[4]
        # if conf < 0.6:
        #     continue
        # if remove_noise_boxes(img, boxes[:4]):
        #     continue
        boxes = boxes[:4]
        boxes = list(map(int, boxes.tolist()))
        save_name = '{}_{}.jpg'.format(name[:-4], idx)
        crop = img[boxes[1]: boxes[3], boxes[0]: boxes[2]]
        cv2.imwrite('../data/crop/private_crop/{}_{}.jpg'.format(name[:-4], idx), crop)
        list_rs.append([save_name, conf, boxes[0], boxes[1], boxes[2], boxes[3]])
    # visualize
    """
    for boxes in bboxes:
        conf =  boxes[4]
        # if boxes[4] < 0.55:
        #     continue
        boxes = boxes[:4]
        boxes = list(map(int, boxes.tolist()))
        img = cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)
        img = cv2.putText(img, str(conf), (boxes[0], boxes[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite('../data/private_visualize/{}'.format(pill_img_path.split('/')[-1]), img)
    """
    return list_rs 

        
if __name__ == '__main__':
    # detect_pill_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/best.pt', force_reload=True)
    detect_pill_model = torch.hub.load('WongKinYiu/yolov7', 'custom', '../checkpoints/best.pt', force_reload=True)
    detect_pill_model.conf = 0.5
    pres_image_test = '../data/private_test/prescription/image'
    pill_image_test = '../data/private_test/pill/image'
    pill_pres_map = '../data/private_test/pill_pres_map.json'
    Path('../data/crop').mkdir(parents=True, exist_ok=True)
    Path('../data/crop/private_crop').mkdir(parents=True, exist_ok=True)
    with open(pill_pres_map, 'r') as fr:
        datas = json.load(fr)
    list_rs = []
    """
    for idx, data in enumerate(datas):
        pres_json_file = data['pres']
        pres_img_name = pres_json_file.replace('.json', '.png')
        pres_img_path = os.path.join(pres_image_test, pres_img_name)
        list_pill_json_file = data['pill']
        for pill_json_file in list_pill_json_file:
            # pill_img_name = pill_json_file.replace('.json', '.jpg')
            
            pill_img_path = os.path.join(pill_image_test, pill_json_file[:-5] + '.jpg')
           
            rs = detect(pill_img_path)
            for r in rs:
                print(r)
                list_rs.append(r)
    """
    for k, v in datas.items():
        for pill_img in v:
            if ' (1)' in pill_img:
                continue
            pill_img_path = os.path.join(pill_image_test, pill_img)
            rs = detect(pill_img_path)
            for r in rs:
                print(r)
                list_rs.append(r)
        
    df = pd.DataFrame(list_rs, columns=['image_name', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])
    df.to_csv('../data/detect_phase_private.csv', index=False)
    
    test_rs = []
    for filename in os.listdir('../data/crop/private_crop'):
        filepath = '../' + 'data/crop/private_crop' + '/' + filename
        test_rs.append([filepath, 0])
    test_df = pd.DataFrame(test_rs, columns=['filepath', 'label'])
    test_df.to_csv('../data/crop/private_crop.csv', index=False)
 
