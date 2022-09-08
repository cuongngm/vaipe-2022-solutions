import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from recog_inference import Stage2
from logger_rewrite import setup_log
logger = setup_log(save_dir='saved/log_107')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
recog_pill_model1 = Stage2(model_name='convnext_base', model_path='weights/convnext_large_384_in22ft1k/fold_0_9.pth', device=device)
recog_pill_model2 = Stage2(model_name='convnext_base', model_path='weights/convnext_large_384_in22ft1k/fold_1_9.pth', device=device)
recog_pill_model3 = Stage2(model_name='convnext_base', model_path='weights/convnext_large_384_in22ft1k/fold_2_9.pth', device=device)
recog_pill_model4 = Stage2(model_name='convnext_base', model_path='weights/convnext_large_384_in22ft1k/fold_3_9.pth', device=device)
recog_pill_model5 = Stage2(model_name='convnext_base', model_path='weights/convnext_large_384_in22ft1k/fold_4_9.pth', device=device)

recog_model_list = [
        recog_pill_model1,
        recog_pill_model2,
        recog_pill_model3,
        recog_pill_model4,
        recog_pill_model5
        ]
recog_model_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

list_rs = []
for filename in tqdm(os.listdir('../data/pill_recog/107')):
    filepath = os.path.join('../data/pill_recog/107', filename)
    print(filepath)
    img = cv2.imread(filepath)
    preds = 0
    for model, w in zip(recog_model_list, recog_model_weights):
        pred = model.recog_pill_pipeline(img)
        pred = torch.softmax(pred, 1)
        pred = pred * w
        preds += pred
    conf_cls = preds.detach().cpu().numpy()
    conf_cls = np.max(conf_cls)
    logger.info('conf_cls = {}'.format(conf_cls))
    print('conf_cls', conf_cls)
    # if conf_cls < 0.9:
    #     continue

    pred_topk = torch.topk(preds, 1)
      
    preds = pred_topk.indices.detach().cpu().numpy()
    preds_str = preds.tolist()
    logger.info('predict = {}'.format(preds_str[0][0]))
    list_rs.append([filepath.replace('../data/pill_recog/', ''), preds_str[0][0], conf_cls])
    
df = pd.DataFrame(list_rs, columns=['image', 'pred_id', 'conf_cls'])
df.to_csv('../data/pseudo_107.csv', index=False)
    
