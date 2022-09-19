# copy and small modify from https://github.com/cubist38/AI4VN_VAIPE/blob/main/evaluate/wmap.py

from eval.wmap import compute_wmap
from eval.csv2coco import *
import pandas as pd
import os
import argparse


def parse():
    parser = argparse.ArgumentParser(description='Evaluation phase')
    parser.add_argument('--pred_path', default='../../submission/6698_trick.csv', type=str)
    parser.add_argument('--gt_path', default='../../submission/gt.csv', type=str)
    return parser.parse_args()


def evaler(results_path: str, train_path: str):
    '''
        Evaluate the results with wmAP metrics
        Args:
            - `results_path`: Path to results file (.csv), with the following columns: `image_name, class_id, confidence_score, x_min, y_min, x_max, y_max`
            - `train_path`: Path to train.csv file (currently in data/tran.csv)
    '''
    results = pd.read_csv(results_path)
    train_df = pd.read_csv(train_path)

    images = results['image_name']
    train_df = train_df[train_df['image_name'].isin(images)]
    print('gt', len(train_df))
    print('pred', len(results))
    anno_path, image_id_map = csv_to_coco(train_df)
    pred_path = results_to_coco(results, image_id_map)
    
    wmap50, wmap = compute_wmap(anno_path, pred_path)
   
    # os.remove(anno_path)
    # os.remove(pred_path)
    return wmap50, wmap

# if __name__ == '__main__':
#     args = parse()
#     wmap50, wmap = eval(args.pred_path, args.gt_path)
#     print('wmAP50:', wmap50)     
#     print('wmAP:', wmap)
