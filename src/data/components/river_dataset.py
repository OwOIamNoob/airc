from typing import Any, Dict, Optional, Tuple
import os 
import numpy as np
import cv2

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import os 
import cv2
import numpy as np
import tqdm

class SpatialRiverDataPool(Dataset):
    def __init__(self, 
                    seed,
                    data_size, 
                    val_box=None, 
                    val_ratio=0.1, 
                    label_ratio=0.8, 
                    patch_size=[512, 512],
                    augment=None,
                    num_workers=4, 
                    num_samples=256):
        self.data_path = data_path
        self.files = []
        self.val_box = val_box
        self.augment = augment
        self.train_path = train_path
        self.val_path = val_path
    
    def setup(self, data_path):
        assert os.path.isdir(data_path), "Path must be dataset directory"
        self.files = sorted(os.listdir(data_path))
    
    def valid(self, pts):
        val = (pts - self.val_box[:2])  / self.val_box[2:]
        return np.any(val < 0) + np.any(val > 1)
    

    def train_crop(self):
        assert os.path.isdir(self.train_path), "Output train folder is unrecognized"
        config = {'unlabeled':[], 'labeled':[]}
        mask = None
        for file in tqdm.tqdm(self.files, desc="Examining label:"):
            data = cv2.imread(file)
            if mask is None:    
                mask = np,zeros_like(data[:, :, 3])
            mask += data[:, :, 3]
            del data
        mask[mask > 0] 
            
    def valid_crop(self):
        assert os.path.isdir(self.val_path), "Output valid folder is unrecognized" 
        