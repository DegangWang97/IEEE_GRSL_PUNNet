import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def map01(img):
    img_01 = (img - img.min())/(img.max() - img.min())
    return img_01

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_map(HSI_old, HSI_new):
    n_row, n_col, n_band = HSI_old.shape
    n_pixels = n_row * n_col
        
    img_olds = np.reshape(HSI_old, (n_pixels, n_band), order='F')
    img_news = np.reshape(HSI_new, (n_pixels, n_band), order='F')        
    sub_img = img_olds - img_news

    detectmap = np.linalg.norm(sub_img, ord = 2, axis = 1, keepdims = True)**2
    detectmap = detectmap/n_band

    # nomalization
    detectmap = map01(detectmap)

    detectmap = np.reshape(detectmap, (n_row, n_col), order='F')
    
    return detectmap

def get_auc(detectmap, gt):
    n_row, n_col = gt.shape
    n_pixels = n_row * n_col
        
    # get auc
    label = np.reshape(gt, (n_pixels,1))
    
    detectmap = np.reshape(detectmap, (n_pixels,1))
    
    auc = roc_auc_score(label, detectmap)
    
    return auc

def TensorToHSI(img):
    HSI = img.squeeze().cpu().data.numpy().transpose((1, 2, 0))
    return HSI
