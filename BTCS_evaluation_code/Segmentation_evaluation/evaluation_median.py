import os
import sys
import time
import warnings
import json

from pathlib import Path
from urllib.parse import urlparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cellpose import io, models, plot, utils
from tqdm import tqdm

from HYT_new import *

# warnings.filterwarnings("ignore")

DIR = "/data/slurm/xuzt/ICA_data/BreaKHis_400X/"
DIR = Path(DIR)
# get list of files to run cellpose on

images = list(DIR.glob("**/*.png"))
images_list = [str(img) for img in images]


def get_groundtruth(img_path):
    model = models.Cellpose(gpu=True, model_type="nuclei")
    img = io.imread(img_path)
    channel = [0, 0]
    masks, flows, styles, diams = model.eval(
        img, diameter=None, channels=channel, invert=True
    )
    np_masks = np.array(masks)
    true = np.int64(np_masks > 0)
    return true


def get_predicted(img_path, User_given):
    pre = nuclei_segementation(
        img_path,
        User_given,
        "one",
        save=False,
        Measure_cells=False,
        Measure_regions=False,
        Calculate_length=False,
        Measure_eccentricity=False,
    )
    predicted = pre.astype(np.int64)
    return predicted


def confusion_matrix(true_mask, pred_mask):
    TP = np.logical_and(true_mask, pred_mask).sum()
    FP = np.logical_and(np.logical_not(true_mask), pred_mask).sum()
    TN = np.logical_and(np.logical_not(true_mask), np.logical_not(pred_mask)).sum()
    FN = np.logical_and(true_mask, np.logical_not(pred_mask)).sum()
    iou = TP / (TP + FP + FN)
    acc = (TP + TN) / (TP + FP + TN + FN)
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    return np.array([[TP, FP], [FN, TN]]), acc, F1, iou


def evaluation(masks_true, masks_predicted):
    cm, acc, F1, iou = confusion_matrix(masks_true, masks_predicted)
    return cm, acc, F1, iou


def get_overall_cm(images_list, user_given):
    results = {}
    # overall_cm = np.zeros((2, 2))
    # overall_acc = 0
    # overall_F1 = 0
    # overall_iou = 0
    for img_path in tqdm(images_list):
        img_name = img_path.split("/")[-1]
        img_type = img_path.split("/")[-2]
        true = get_groundtruth(img_path)
        predicted = get_predicted(img_path, User_given=user_given)
        cm, acc, F1, iou = evaluation(true, predicted)
        # overall_cm += cm
        # overall_acc += acc
        # overall_iou += iou
        re = {img_name: {'type':img_type, 'cm': cm.tolist(), 'acc': acc, 'F1': F1, 'iou': iou}}
        print(re)
        results.update(re)
    # overall_acc /= len(images_list)
    # overall_F1 /= len(images_list) - ill_F1
    # overall_iou /= len(images_list)
    # results.update(re)
    return results

results = get_overall_cm(images_list, 'median')
with open('/data/slurm/xuzt/ICA_data/evaluation_results_median.json', 'w') as f:
    json.dump(results, f)

# overall_cm, overall_acc, overall_F1, overall_iou = get_overall_cm(images_list, "few")

# print("overall_cm: ", overall_cm)
# print("overall_acc: ", overall_acc)
# print("overall_F1: ", overall_F1)
# print("overall_iou: ", overall_iou)

# results = f"overall_cm:{overall_acc}, overall_acc:{overall_acc}, overall_F1:{overall_F1}, overall_iou:{overall_iou}"

# with open("./results.txt", "w") as f:
#     f.write(results)
#     f.close()
