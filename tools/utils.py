import torch
import random
import logging
import numpy as np
from fpointnet_train import provider_fpointnet as provider

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3

MEAN_SIZE_ARR = np.array([
    [ 4.8, 1.8, 1.5],
    [10.0, 2.6, 3.2],
    [ 2.0, 1.0, 1.6],
])

class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    ENDC = '\033[0m'

def fixSeed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def reorganize_info(infos):
    new_info = {}
    for info in infos:
        token = info['token']
        new_info[token] = info
    return new_info

def angle2class(angle, num_class):
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle

def size2class(lwh):
    diff = lwh[np.newaxis, ...] - MEAN_SIZE_ARR
    diff = np.linalg.norm(diff, axis=1)
    class_id = np.argmin(diff)
    residual_size = lwh - MEAN_SIZE_ARR[class_id]
    return class_id, residual_size

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle

def class2size(pred_cls, residual):
    mean_size = MEAN_SIZE_ARR[pred_cls]
    return mean_size + residual

def compute_box3d_iou(center_pred, heading_logits, heading_residuals, size_logits, size_residuals, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label):
    bs = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)
    heading_residual = np.array([heading_residuals[i, heading_class[i]] for i in range(bs)])
    size_class = np.argmax(size_logits, 1)
    size_residual = np.vstack([size_residuals[i, size_class[i], :] for i in range(bs)])

    iou2d_list = []
    iou3d_list = []
    for i in range(bs):
        heading_angle = class2angle(heading_class[i], heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = provider.get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i], heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = provider.get_3d_box(box_size_label, heading_angle_label, center_label[i])

        iou_3d, iou_2d = provider.box3d_iou(corners_3d, corners_3d_label)
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    
    return np.array(iou2d_list, dtype=np.float32), np.array(iou3d_list, dtype=np.float32)