from typing import Any, Dict, Optional, Tuple
import os 
import tqdm

import numpy as np
import cv2

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import rootutils 
rootutils.setup_root(__file__, indicator="setup.py", pythonpath=True)
from src.data.components.transforms import *



class RiverDataPool(Dataset):
    def __init__(self, 
                    data_path, 
                    train_path,
                    val_path, 
                    seed=0,
                    sample_size=256,
                    label_ratio=0.8, 
                    patch_size=[512, 512],
                    augment=None,
                    num_workers=4, 
                    temporal=False,
                    val_center=None, 
                    val_ratio=0.1,
                    max_samples_per_class=None):
        # File structures
        self.data_path = data_path
        self.train_path = train_path
        self.val_path = val_path
        self.files = []

        # Configuration
        self.augment = augment
        self.patch_size = patch_size
        self.sample_size = sample_size
        self.temporal = temporal
        self.label_ratio = label_ratio
        self.num_workers = num_workers
        self.generator = np.random.RandomState(seed)

        # Validation data config
        self.val_center = val_center
        self.val_ratio = val_ratio
        
        # Post-config properties
        self.spatial_size = None
        self.fg_indices = None
        self.bg_indices = None
        self.setup(data_path, max_samples_per_class=max_samples_per_class)
    
    def setup(self, data_path, max_samples_per_class=None):
        assert os.path.isdir(data_path), "Path must be dataset directory"
        if data_path != self.data_path:
            self.data_path = data_path

        self.files = sorted(os.listdir(data_path))
        if not os.path.isdir(self.train_path):
            os.mkdir(self.train_path)
        if not os.path.isdir(self.val_path):
            os.mkdir(self.val_path)
        # prepare cut string
        self.get_indices(max_samples_per_class=max_samples_per_class)

    def get_indices(self, max_samples_per_class: int | None = None):
        if self.bg_indices is None or self.fg_indices is None:
            mask = None
            for file in tqdm.tqdm(self.files, desc="Examining label:"):
                data = cv2.imread(file)
                if mask is None:
                    mask = np.zeros_like(data[:, :, 3])
                mask += data[:, :, 3]
                del data
            mask[mask > 0] = 1
            self.spatial_size = mask.shape[:2]
            val_bbox = np.multiply(np.sqrt(self.val_ratio), mask.shape[:, 2])
            if self.val_center is None:
                choices = np.array(np.where(mask == 1)).T
                self.val_center = choices[np.random.randint(0, choices.shape[0])]
                self.val_center = correct_crop_center(self.val_center,  val_bbox, mask.shape[:2], allow_smaller=False)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, [patch_size[0] // 2, patch_size[1] // 2]), 1)
            mask[max(0, self.val_center[0] - (val_bbox[0] + self.patch_size[0])//2):
                    min(mask.shape[0], self.val_center[0] + (val_bbox[0] + self.patch_size[0])//2),
                    max(0, self.val_center[1] - (val_bbox[1] + self.patch_size[1])//2): 
                    min(mask.shape[1], self.val_center[1] + (val_bbox[1] + self.patch_size[1])//2)] = 2 
            [self.bg_indices, self.fg_indices] = map_classes_to_indices(mask, [0, 1], max_samples_per_class=max_samples_per_class)


    def rect_crop(self):
        assert os.path.isdir(self.train_path), "Output train folder is unrecognized"
        assert os.path.isdir(self.val_path), "Output valid folder not found"
        cfg = []  
        sup = [False] * self.sample_size
        val_bbox = np.multiply(np.sqrt(self.val_ratio), mask.shape[:, 2])
        for idx, file in enumerate(tqdm.tqdm(self.files, desc="Generating training set")):
            # Prepare 
            year = file.split(".")[0]
            img_dir = os.path.join(self.train_path, year) if self.temporal else os.path.join(self.train_path, 'training')
            if os.path.isdir(img_dir):
                subprocess.run('rm', '-rf', path)
            os.mkdir(img_dir)
            image = cv2.imread(os.path.join(self.data_path, file))
            # Valid crop
            cv2.imwrite(os.path.join(self.val_path, f"{year}.png"), image[  max(0, self.val_center[0] - val_bbox[0]//2):min(mask.shape[0], self.val_center[0] + val_bbox[0]//2),
                                                                            max(0, self.val_center[1] - val_bbox[1]//2):min(mask.shape[1], self.val_center[1] + val_bbox[1]//2)])
            # Generating training patches' centers
            if not self.temporal or centers is None:
                centers = generate_pos_neg_label_crop_centers(  self.patch_size, 
                                                        self.sample_size, 
                                                        pos_ratio=self.label_ratio,
                                                        label_spatial_shape=self.spatial_size,
                                                        fg_indices=self.fg_indices,
                                                        bg_indices=self.bg_indices,
                                                        allow_smaller=False)
            # Crop training patches
            for index, center in enumerate(centers):
                cropped = image[center[0] - patch_size[0]//2:center[0] + patch_size[0]//2, 
                                center[1] - patch_size[1]//2:center[1] + patch_size[1]//2]
                id = index if self.temporal else self.sample_size * idx + index
                cv2.imwrite(os.path.join(img_dir, f"{id}.png"), cropped)
            


    def augment_crop(self):
        assert os.path.isdir(self.train_path), "Output train folder is unrecognized"
        assert os.path.isdir(self.val_path), "Output valid folder not found"
        cfg = []
        profiles, centers = None    
        sup = [False] * self.sample_size
        val_bbox = np.muliply(np.sqrt(self.val_ratio), mask.shape[:, 2])
        header = {'h': self.patch_size[0], 'w': self.patch_size[1]}
        for file in tqdm.tqdm(self.files, desc="Generating training set"):
            year = file.split(".")[0]
            img_dir = os.path.join(self.train_path, year) if self.temporal else os.path.join(self.train_path, 'training')
            if os.path.isdir(img_dir):
                subprocess.run('rm', '-rf', path)
            os.mkdir(img_dir)
            image = cv2.imread(os.path.join(self.data_path, file))
            cv2.imwrite(os.path.join(self.val_path, f"{year}.png"), image[  max(0, self.val_center[0] - val_bbox[0]//2):min(mask.shape[0], self.val_center[0] + val_bbox[0]//2),
                                                                            max(0, self.val_center[1] - val_bbox[1]//2):min(mask.shape[1], self.val_center[1] + val_bbox[1]//2)])
            
            if self.temporal:
                #(data, gnt, fg_indices, bg_indices, header, None, profiles=profiles, centers=centers)
                patches, profiles, centers = transform.cut_centers_transform(image, 
                                                                            gnt=self.generator, 
                                                                            fg_indices=self.fg_indices, 
                                                                            bg_indices=self.bg_indices, 
                                                                            header=header, 
                                                                            augment=self.augment, 
                                                                            profiles=profiles,
                                                                            centers=centers,
                                                                            pos_ratio=self.label_ratio,
                                                                            num_workers=self.num_workers)
            else:
                #(data, gnt, fg_indices, bg_indices, header, None, profiles=profiles, centers=centers)
                patches, _, _ = transform.cut_centers_transform(image, 
                                                                gnt=self.generator, 
                                                                fg_indices=self.fg_indices, 
                                                                bg_indices=self.bg_indices, 
                                                                header=header, 
                                                                augment=self.augment, 
                                                                profiles=profiles,
                                                                centers=centers,
                                                                pos_ratio=self.label_ratio,
                                                                num_workers=self.num_workers)
            for index, patch in enumerate(patches):
                id = index if self.temporal else self.sample_size * idx + index
                cv2.imwrite(os.path.join(img_dir, f"{id}.png"), patch)
    
if __name__ == "__main__":
    root = "/work/hpc/potato/airc/data"
    data_path = os.path.join(root, "v2")
    train_path = os.path.join(root, "dataset/v1/train")
    val_path = os.path.join(root, "dataset/v1/val")


            