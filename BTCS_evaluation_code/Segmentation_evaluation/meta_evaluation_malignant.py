import json
import warnings
from pathlib import Path

import cv2
import matplotlib as mpl

# import time, os, sys
import matplotlib.pyplot as plt
import numpy as np
from cellpose import io, models, plot, utils
from PIL import Image
from tqdm import tqdm



warnings.filterwarnings("ignore")

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



def get_datasets(human_data_dir, model_data_dir):
    # human_dir = '/data/slurm/xuzt/ICA_data/PS-Transparent'
    human_dir = Path(human_data_dir)
    human_ma_path = human_dir / "Malignant"
    human_images = list(human_ma_path.glob("*.png"))
    human_images_list = [str(img) for img in human_images]
    human_image_name_list = [img.split("/")[-1] for img in human_images_list]
    human_image_name_list_dic = {
        human_image_name_list[i]: human_images_list[i]
        for i in range(len(human_image_name_list))
    }

    model_dir = Path(model_data_dir)
    model_ma_dir = model_dir / "malignant"
    model_images = list(model_ma_dir.glob("*.png"))
    model_images_list = [str(img) for img in model_images]
    model_image_name_list = [img.split("/")[-1] for img in model_images_list]
    model_image_name_list_dic = {
        model_image_name_list[i]: model_images_list[i]
        for i in range(len(model_image_name_list))
    }
    model_image_name_list_dic_new = {
        k: model_image_name_list_dic[k] for k in human_image_name_list_dic.keys()
    }
    return human_image_name_list_dic, model_image_name_list_dic_new


def generate_masks_for_human_labels(img_path):
    print("======== Begin generating masks for artificial segmentation ========")
    with Image.open(img_path) as image:
        pixel_array = np.array(image)
        p1 = pixel_array[:, :, 3]
        black_pixel_data = np.where(p1 > 0, 1, 0)

    # filled with masks
    mask = black_pixel_data.astype(np.uint8)
    # Find contours from the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the filled contours
    filled_mask = np.zeros_like(mask)

    # Draw filled contours on the empty mask
    cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
    return filled_mask


def generate_masks_for_model_prediction(img_path):
    print("======== Begin generating masks for model prediction ========")
    model = models.Cellpose(gpu=True, model_type="nuclei")
    img = io.imread(img_path)
    channel = [0, 0]
    masks, flows, styles, diams = model.eval(
        img, diameter=None, channels=channel, invert=True
    )
    np_masks = np.array(masks)
    true = np.int64(np_masks > 0)
    return true


def generate_list_of_masks_for_human_labels(img_path_list):
    print(
        "======== Begin generating list of masks for artificial segmentation ========"
    )
    filled_mask_list = []
    for img_path in tqdm(img_path_list):
        img_name = img_path.split("/")[-1]
        filled_mask = generate_masks_for_human_labels(img_path)
        filled_mask_list.append((img_name, filled_mask))
    return filled_mask_list


def generate_list_of_masks_for_model_prediction(img_path_list):
    print("======== Begin generating list of masks for model prediction ========")
    predicted_mask_list = []
    for img_path in tqdm(img_path_list):
        img_name = img_path.split("/")[-1]
        predicted_mask = generate_masks_for_model_prediction(img_path)
        predicted_mask_list.append((img_name, predicted_mask))
    return predicted_mask_list


def get_evaluation_results_for_single_picture(human_masks, model_masks):
    human_img_name, Human_masks = human_masks
    _, Model_masks = model_masks
    cm, acc, F1, iou = evaluation(Human_masks, Model_masks)
    print(
        f"For {human_img_name}, the confusion matrix is \n{cm} \nthe accuracy is {acc} \nthe F1 score is {F1} \nthe IoU is {iou}"
    )
    return {human_img_name: {"cm": cm.tolist(), "acc": acc, "F1": F1, "iou": iou}}


def get_evaluation_results_for_all_pictures(human_masks_list, model_masks_list):
    print("======== Begin evaluation ========")
    # cm_list = []
    # acc_list = []
    # F1_list = []
    # iou_list = []
    results = {}
    for i in tqdm(range(len(human_masks_list))):
        single_result = get_evaluation_results_for_single_picture(
            human_masks_list[i], model_masks_list[i]
        )
        results.update(single_result)
    return results


human_data_dir = "/data/slurm/xuzt/ICA_data/PS-Transparent"
model_data_dir = "/data/slurm/xuzt/ICA_data/BreaKHis_400X/train/"
human_image_name_list_dic, model_image_name_list_dic = get_datasets(
    human_data_dir, model_data_dir
)
picture_list = human_image_name_list_dic.keys()
print("======== Begin generating masks for human_labels ========")
human_masks_list = generate_list_of_masks_for_human_labels(
    [human_image_name_list_dic[img] for img in picture_list]
)
print("======== Begin generating masks for model_prediction ========")
model_masks_list = generate_list_of_masks_for_model_prediction(
    [model_image_name_list_dic[img] for img in picture_list]
)
results = get_evaluation_results_for_all_pictures(
    human_masks_list, model_masks_list
)

print("======== Begin saving results ========")
re = json.dumps(results)
with open("/data/slurm/xuzt/ICA_data/evaluation_results.json", "w") as f:
    f.write(re)
print("======== Done ========")

