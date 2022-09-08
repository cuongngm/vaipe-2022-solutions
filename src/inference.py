import os
import torch
import json
import cv2
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from recog_inference import Stage2
from logult import setup_log
from util import remove_noise_boxes


def parse_args():
    parser = argparse.ArgumentParser(description='Hyper_parameter')
    parser.add_argument('--detect_model_path', type=str, default=None, help='detect model path')
    parser.add_argument('--recog_model_path', type=str, default=None, help='recog model path')
    parser.add_argument('--log_path', type=str, default='saved/exp1', help='log path')
    args = parser.parse_args()
    return args


def run_pipeline(pill_img_path, pres_img_name, detect_pill_model, recog_model_list, recog_model_weights, drug_mapping, logger): 
    all_info = []
    pill_id_in_pres = drug_mapping[pres_img_name.replace('.png', '')]
    pill_id_in_pres = list(map(int, pill_id_in_pres))
    print('pill_id_in_pres', pill_id_in_pres)
    print('pill_img_path', pill_img_path)
    logger.info('pill_img_path {}'.format(pill_img_path))
    logger.info('pill_id_in_pres {}'.format(pill_id_in_pres))
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
        preds = 0
        for model, w in zip(recog_model_list, recog_model_weights):
            pred = model.recog_pill_pipeline(crop)
            pred = torch.softmax(pred, 1)
            pred = pred * w
            # pred = pred.to(device2)
            preds += pred
        
        """
        pred1_id = recog_pill_model1.recog_pill_pipeline(crop)
        pred1_id = torch.softmax(pred1_id, 1)
        # pred2_id = 0
        # pred3_id = 0
        if recog_pill_model2 is not None:
            pred2_id = recog_pill_model2.recog_pill_pipeline(crop)
            pred2_id = torch.softmax(pred2_id, 1)
        if recog_pill_model3 is not None:
            pred3_id = recog_pill_model3.recog_pill_pipeline(crop)
            pred3_id = torch.softmax(pred3_id, 1)
        pred_id = pred1_id * w_ensemble[0] + pred2_id * w_ensemble[1] + pred3_id * w_ensemble[2]
        """
        conf_cls = preds.detach().cpu().numpy()
        conf_cls = np.max(conf_cls)
        list_conf.append(conf_cls)
        
        logger.info('conf_cls {}'.format(conf_cls))
        pred_topk = torch.topk(preds, 2)
        
        preds = pred_topk.indices.detach().cpu().numpy()
        preds_str = preds.tolist()
        preds_str = list(map(str, preds_str))
        preds_str = ','.join(preds_str)
        
        list_pred_pill.append(preds_str)
        check = 0
        for pred in preds[0]:
            if pred in pill_id_in_pres:
                check = 1
                pred_id = pred
                break
        if check == 0:
            pred_id = 107
        # if conf_cls < 0.75:
        #     pred_id = 107
        logger.info('pred_id {}'.format(pred_id))
        logger.info('====================')
        all_info.append([pred_id, conf_cls, boxes[0], boxes[1], boxes[2], boxes[3]])
    """
    # visualize
    for boxes, pred, conf in zip(bboxes, list_pred_pill, list_conf):
        if boxes[4] < 0.55:
            continue
        boxes = boxes[:4]
        boxes = list(map(int, boxes.tolist()))
        img = cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)
        img = cv2.putText(img, pred, (boxes[2], boxes[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        img = cv2.putText(img, str(conf), (boxes[0], boxes[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite('test/{}'.format(pill_img_path.split('/')[-1]), img)
    """
    return all_info


if __name__ == '__main__':
    opt = parse_args()
    logger = setup_log(save_dir=opt.log_path)
    device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    detect_pill_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/best.pt', force_reload=True)
    # recog_pill_model1 = Stage2(model_name='convnext_base', model_path='weights/convnext_base_384_in22ft1k/fold0.pth', device=device2)
    # recog_pill_model2 = Stage2(model_name='convnext_base', model_path='weights/convnext_large_384_in22ft1k/fold_0_9.pth', device=device2)
    # recog_pill_model3 = Stage2(model_name='convnext_base', model_path='weights/convnext_xlarge_384_in22ft1k/fold0.pth', device=device2)
    # recog_pill_model4 = Stage2(model_name='convnext_base', model_path='weights/convnext_large_384_in22ft1k/fold_3_9.pth', device=device2)
    recog_pill_model4 = Stage2(model_name='convnext_xlarge', model_path='weights/convnext_xlarge_384_in22ft1k/fold0.pth', device=device1)
    recog_pill_model5 = Stage2(model_name='convnext_xlarge', model_path='weights/convnext_xlarge_384_in22ft1k/fold1.pth', device=device1)
    # recog_pill_model4 = Stage2(model_name='efficientnet', model_path='weights/tf_efficientnet_b7_ns/fold_1_9.pth', device=device1)
    # recog_pill_model5 = Stage2(model_name='efficientnet', model_path='weights/tf_efficientnet_b7_ns/fold_0_9.pth', device=device1)
    # recog_pill_model5 = Stage2(model_name='vit_large_patch16_384', model_path='weights/vit_large_patch16_384/fold0.pth', device=device2)
    
    recog_model_list = [
            # recog_pill_model1,
            # recog_pill_model2,
            # recog_pill_model3,
            recog_pill_model4,
            recog_pill_model5
            ]
    recog_model_weights = [0.5, 0.5]
    assert len(recog_model_list) == len(recog_model_weights), 'list weight not compatible with list model'
    # recog_pill_model3 = None
    # recog_pill_model3 = Stage2(model_name='convnext_base', model_path='weights/convnext_2fold/fold_1_9.pth')

    logger.info('Code chay inference bai vaipe')
    logger.info('Model su dung')
    
    pres_image = '../data/public_train/prescription/image'
    pres_label = '../data/public_train/prescription/label'
    pill_image = '../data/public_train/pill/image'
    pill_label = '../data/public_train/pill/label'

    pres_image_test = '../data/public_test_new/prescription/image'
    pill_image_test = '../data/public_test_new/pill/image'

    pill_pres_map = '../data/public_test_new/pill_pres_map.json'
    # drug_mapping = '../data/giang/drug_new.npy'
    # drug_mapping = np.load(drug_mapping, allow_pickle=True)
    # drug_mapping = drug_mapping.tolist()
    with open('../data/cuong_mapping.json', 'r') as fr:
        drug_mapping = json.load(fr)
    # pill_img_path = '../data/public_test/pill/image/VAIPE_P_59_60.jpg'
    # pres_img_name = 'VAIPE_P_TEST_59.png'
    # all_infos = run_pipeline(pill_img_path, pres_img_name, detect_pill_model, recog_pill_model1, recog_pill_model2, recog_pill_model3, drug_mapping, logger)
    
    list_rs = []
    with open(pill_pres_map, 'r') as fr:
        datas = json.load(fr)
    for data in datas:
        pres_json_file = data['pres']
        pres_img_name = pres_json_file.replace('.json', '.png')
        pres_img_path = os.path.join(pres_image_test, pres_img_name)
        list_pill_json_file = data['pill']
        for pill_json_file in list_pill_json_file:
            # pill_img_name = pill_json_file.replace('.json', '.jpg')
            pill_img_name = pill_json_file + '.jpg'
            pill_img_path = os.path.join(pill_image_test, pill_img_name)
            all_infos = run_pipeline(pill_img_path, pres_img_name, detect_pill_model, recog_model_list, recog_model_weights, drug_mapping, logger)

            for info in all_infos:
                pred_id = info[0]
                pred_conf = info[1]
                xmin = info[2]
                ymin = info[3]
                xmax = info[4]
                ymax = info[5]
                list_rs.append([pill_img_name, pred_id, pred_conf, xmin, ymin, xmax, ymax])
    logger.info('End...........................')
    df = pd.DataFrame(list_rs, columns=['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])
    # image_name, class_id, confidence_score, xmin, ymin, xmax, ymax
    df.to_csv('results.csv', index=False)
    
