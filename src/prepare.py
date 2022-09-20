import os
import cv2
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from pathlib import Path
from util import xywh2yolo
from tqdm import tqdm
import multiprocessing
from util import create_drugdict


for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def get_pres_yolo(pres_label):
    # generate box of pillname in prescription image and save them in data/pres_yolo folder
    Path('../data/pres_yolo/images').mkdir(parents=True, exist_ok=True)
    Path('../data/pres_yolo/labels').mkdir(parents=True, exist_ok=True)
    for labelname in tqdm(os.listdir(pres_label)):
        labelpath = os.path.join(pres_label, labelname)
        imgpath = os.path.join(pres_image, labelname.replace('.json', '.png'))
        img_cv2 = cv2.imread(imgpath)
        height, width = img_cv2.shape[:2]
        shutil.copy(imgpath, os.path.join('../data/pres_yolo/images', labelname.replace('.json', '.jpg')))
        f = open('../data/pres_yolo/labels/' + labelname.replace('.json', '.txt'), 'w+')
        with open(labelpath, 'r') as fr:
            pres_infos = json.load(fr)
            for pres_info in pres_infos:
                if pres_info['label'] == 'drugname':
                    bbox = pres_info['box']
                    center_x = ((bbox[0] + bbox[2]) / 2) / width
                    center_y = ((bbox[1] + bbox[3]) / 2) / height
                    w = (bbox[2] - bbox[0]) / width
                    h = (bbox[3] - bbox[1]) / height
                    f.write('0 ' + str(center_x) + ' ' + str(center_y) + ' ' + str(w) + ' ' + str(h) + '\n')
        f.close()
        

def get_pill_yolo(pill_label):
    # generate box of pill in pill image and save them in data/pill_yolo folder
    # single class
    for labelname in tqdm(os.listdir(pill_label)):
        labelpath = os.path.join(pill_label, labelname)
        # print(labelpath)
        imgpath = os.path.join(pill_image, labelname.replace('.json', '.jpg'))
        shutil.copy(imgpath, os.path.join('../data/pill_yolo_1class/images', labelname.replace('.json', '.jpg')))
        img = Image.open(imgpath)
        w_img, h_img = exif_size(img)
        all_info = '' 
        with open(labelpath, 'r') as fr:
            pill_infos = json.load(fr)
            for idx, pill_info in enumerate(pill_infos):
                x = pill_info['x']
                y = pill_info['y']
                h = pill_info['h']
                w = pill_info['w']
                pill_id = pill_info['label']
                pill_path =  '../data/pill_recog/' + str(pill_id)
                x_center, y_center, w_norm, h_norm = xywh2yolo(x, y, w, h, w_img, h_img)
                # all_info += str(pill_id) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w_norm) + ' ' + str(h_norm) + '\n'
                all_info += '0' + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w_norm) + ' ' + str(h_norm) + '\n'
        with open(os.path.join('../data/pill_yolo_1class/labels', labelname.replace('.json', '.txt')), 'w') as fw:
            fw.write(all_info)


def get_pill_recog(pill_image, pill_label):
    mode = pill_image.split('_')[-1]
    # generate crop pill data and save them in pill_recog/ folder
    for labelname in tqdm(os.listdir(pill_label)):
        labelpath = os.path.join(pill_label, labelname)
        # print(labelpath)
        imgpath = os.path.join(pill_image, labelname.replace('.json', '.jpg'))
        # shutil.copy(imgpath, os.path.join('../data/pill_yolo/images', labelname.replace('.json', '.jpg')))
        # img = Image.open(imgpath)
        img_cv2 = cv2.imread(imgpath)
        # w_img, h_img = exif_size(img)
        all_info = ''
        with open(labelpath, 'r') as fr:
            pill_infos = json.load(fr)
            for idx, pill_info in enumerate(pill_infos):
                x = pill_info['x']
                y = pill_info['y']
                h = pill_info['h']
                w = pill_info['w']
                img_crop = img_cv2[y:y+h, x:x+w]
                # img_crop = img.crop((x, y, x+w, y+h))
                pill_id = pill_info['label']
                pill_path =  '../data/crop/train_crop/'.format(mode) + str(pill_id)
                if not os.path.exists(pill_path):
                    Path(pill_path).mkdir(parents=True, exist_ok=True)
                # img_crop.save('../data/pill_recog/{}/{}_pill{}.jpg'.format(str(pill_id), labelname.replace('.json', ''), idx))
                cv2.imwrite('../data/crop/train_crop/{}/{}_pill{}.jpg'.format(mode, str(pill_id), labelname.replace('.json', ''), idx), img_crop)
                # x_center, y_center, w_norm, h_norm = xywh2yolo(x, y, w, h, w_img, h_img)
                # all_info += str(pill_id) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w_norm) + ' ' + str(h_norm) + '\n'
        # with open(os.path.join('../data/pill_yolo/labels', labelname.replace('.json', '.txt')), 'w') as fw:
        #     fw.write(all_info)


def get_pill_recog_val(val_csv):
    df = pd.read_csv(val_csv)
    saved = '../data/crop/val_crop/'
    root = '../data/public_val/pill/image/'
    for idx in tqdm(range(len(df))):
        class_id = df.loc[idx, 'class_id']
        if class_id == 107:
            continue
        imagename = df.loc[idx, 'image_name']
        path = root + imagename
        img = cv2.imread(path)
        xmin = df.loc[idx, 'x_min']
        ymin = df.loc[idx, 'y_min']
        xmax = df.loc[idx, 'x_max']
        ymax = df.loc[idx, 'y_max']
        crop = img[ymin:ymax, xmin:xmax]
        Path(saved + str(class_id)).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(saved + str(class_id), imagename[:-4] + '_' + str(idx) + '.jpg'), crop)
        
        
def get_pres_recog(pres_label):
    # generate drugname in prescription image to training ocr pillname
    Path('../data/ocr/pres_text_crop').mkdir(parents=True, exist_ok=True)
    all_info = ''
    list_dict_drug = []
    for labelname in tqdm(os.listdir(pres_label)):
        labelpath = os.path.join(pres_label, labelname)
        imgpath = os.path.join(pres_image, labelname.replace('.json', '.png'))
        print(imgpath)
        img_cv2 = cv2.imread(imgpath)
        with open(labelpath, 'r') as fr:
            pres_infos = json.load(fr)
            for idx, pres_info in enumerate(pres_infos):
                box = pres_info['box']
                text = pres_info['text']
                xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                img_crop = img_cv2[ymin:ymax, xmin:xmax]
                cv2.imwrite('../data/pres_text_crop/{}_{}.jpg'.format(labelname.replace('.json', ''), idx), img_crop)
                all_info += '../data/pres_text_crop/{}_{}.jpg'.format(labelname.replace('.json', ''), idx) + '\t' + text + '\n'
                if 'mapping' in pres_info.keys():
                    drug_id = pres_info['mapping']
                    drug_name = pres_info['text']
                    dict_drug = dict()
                    dict_drug['drug_id'] = drug_id
                    dict_drug['drug_name'] = drug_name
                    list_dict_drug.append(dict_drug)
        # break
    with open('../data/ocr/pres_text.txt', 'w') as fw:
        fw.write(all_info)
    with open('../data/drug_dict.json', 'w') as fw_j:
        json.dump(list_dict_drug, fw_j)
        

def gen_datacsv_model_extract():
    root = '../data/crop'
    train = 'train_crop'
    val = 'val_crop'
    test = 'test_crop'
    
    print('train!!')
    train_rs = []
    for subdir in os.listdir(os.path.join(root, train)):
        label = subdir
        if label == '107':
            continue
        for filename in os.listdir(os.path.join(root, train, subdir)):
            filepath = os.path.join(root, train, subdir) + '/' + filename
            train_rs.append([filepath, label])
    train_df = pd.DataFrame(train_rs, columns=['filepath', 'label'])
    train_df.to_csv('../data/crop/train_crop.csv', index=False)

    print('val!!')
    val_rs = []
    for subdir in os.listdir(os.path.join(root, val)):
        label = subdir
        for filename in os.listdir(os.path.join(root, val, subdir)):
            filepath = os.path.join(root, val, subdir) + '/' + filename
            val_rs.append([filepath, label])
    val_df = pd.DataFrame(val_rs, columns=['filepath', 'label'])
    val_df.to_csv('../data/crop/val_crop.csv', index=False)
    """ 
    print('test!!')
    test_rs = []
    for filename in os.listdir(os.path.join(root, test)):
        filepath = os.path.join(root, test) + '/' + filename
        test_rs.append([filepath, 0])
    test_df = pd.DataFrame(test_rs, columns=['filepath', 'label'])
    test_df.to_csv('../data/crop/test_crop.csv', index=False)
    """
    
if __name__ == '__main__':
    # task 1
    pres_image = '../data/public_train/prescription/image'
    pres_label = '../data/public_train/prescription/label'
    pill_image_train = '../data/public_train/pill/image'
    pill_label_train = '../data/public_train/pill/label'
    pill_image_val = '../data/public_train/pill/image'
    pill_label_val = '../data/public_train/pill/label'
    pill_image_test = '../data/public_train/pill/image'
    pill_label_test = '../data/public_train/pill/label'

    Path('../data/pill_yolo_1class/images').mkdir(parents=True, exist_ok=True)
    Path('../data/pill_yolo_1class/labels').mkdir(parents=True, exist_ok=True)

    Path('../data/pill_recog').mkdir(parents=True, exist_ok=True)
    print('get pill yolo')
    # get_pill_yolo(pill_label_train)
    print('get pres yolo')
    # get_pres_yolo(pres_label)
    print('get pill recog train')
    get_pill_recog(pill_image_val, pill_label_val)
    print('get pill recog val')
    get_pill_recog_val('../data/public_val/pubval_groundtruth.csv')
    print('get pres recog')
    # get_pres_recog(pres_label)
    print('create drugname dict')
    # create_drugdict()
    print('gen csv data')
    gen_datacsv_model_extract()
