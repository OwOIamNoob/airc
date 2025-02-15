from typing import Any, Dict, Optional, Tuple
from collections.abc import Sequence, Callable
import os 
import tqdm
import pickle
from functools import partial

import numpy as np
import cv2

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

import rootutils 
rootutils.setup_root(__file__, indicator="setup.py", pythonpath=True)
from src.data.components.transforms import array, transform
from src.utils.dir import clear_directory

def collate_batch(list_of_dict, post_act: Callable | None = None):
    output = {}
    for entity in list_of_dict: 
        for key in entity.keys():
            if key not in output.keys():
                output[key] = []
            output[key].append(entity[key])

    if post_act:
        for key in output.keys():
            output[key] = post_act(output[key])
    return output

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
                    temporal=False,
                    val_center=None, 
                    val_ratio=0.1,
                    checkpoint_path=None,
                    max_samples_per_class=None,
                    num_workers=4, 
                    deterministic=False):
        # File structures
        self.data_path = data_path
        self.train_path = train_path
        self.val_path = val_path
        self.branches = ['unsup', 'sup']
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
        self.deterministic = deterministic
        self.fg_indices = None
        self.bg_indices = None
        self.profiles = []
        self.centers = []
        self.labels = []
        self.setup(data_path, max_samples_per_class=max_samples_per_class)
    

    def load_ckpt(self, path):
        if path is not None:
            """ Load ckpt from npz file, will force temporal to ckpt shift"""
            data = np.load(path, allow_pickle=True)
            self.temporal = data['temporal']
            self.profiles = data['profiles']
            self.centers = data['centers']
            self.labels = data['labels']

    def setup(self, data_path, max_samples_per_class=None):
        assert os.path.isdir(data_path), "Path must be dataset directory"
        if data_path != self.data_path:
            self.data_path = data_path

        self.files = sorted(os.listdir(data_path))
        if not self.temporal:
            self.profiles = [[]] * len(self.files)
            self.centers = [[]] * len(self.files)
            self.labels = [[]] * len(self.files)
        if not os.path.isdir(self.train_path):
            os.mkdir(self.train_path)
        if not os.path.isdir(self.val_path):
            os.mkdir(self.val_path)
        
        for branch in self.branches:
            path = os.path.join(self.train_path, branch)
            if not os.path.isdir(path):
                os.mkdir(path)
            else: 
                clear_directory(path)
        # prepare cut string
        self.get_indices(max_samples_per_class=max_samples_per_class)

    def get_indices(self, max_samples_per_class: int | None = None):
        if self.bg_indices is None or self.fg_indices is None:
            mask = None
            for file in tqdm.tqdm(self.files, desc="Examining label:"):
                data = cv2.imread(os.path.join(self.data_path, file), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    mask = np.zeros_like(data[:, :, 3])
                mask += data[:, :, 3]
                del data
            mask[mask > 0] = 1
            self.spatial_size = mask.shape[:2]
            val_bbox = np.multiply(np.sqrt(self.val_ratio), mask.shape[:2]).astype(int)
            if self.val_center is None:
                choices = np.array(np.where(mask == 1)).T
                self.val_center = choices[np.random.randint(0, choices.shape[0])]
                print(self.val_center, val_bbox)
                self.val_center = array.correct_crop_centers(self.val_center,  val_bbox, mask.shape[:2], allow_smaller=False)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, [self.patch_size[0] // 2, self.patch_size[1] // 2]), 1)
            mask[max(0, self.val_center[0] - (val_bbox[0] + self.patch_size[0])//2):
                    min(mask.shape[0], self.val_center[0] + (val_bbox[0] + self.patch_size[0])//2),
                    max(0, self.val_center[1] - (val_bbox[1] + self.patch_size[1])//2): 
                    min(mask.shape[1], self.val_center[1] + (val_bbox[1] + self.patch_size[1])//2)] = 2 
            [self.bg_indices, self.fg_indices] = array.map_classes_to_indices(mask, [0, 1], max_samples_per_class=max_samples_per_class)


    def rect_crop(self):
        for branch in self.branches:
            path = os.path.join(self.train_path, branch)
            if not os.path.isdir(path):
                os.mkdir(path)
            else: 
                clear_directory(path)
        
        assert os.path.isdir(self.train_path), "Output train folder is unrecognized"
        assert os.path.isdir(self.val_path), "Output valid folder not found"
        cfg = []  
        val_bbox = np.multiply(np.sqrt(self.val_ratio), self.spatial_size).astype(int)
        for idx, file in enumerate(tqdm.tqdm(self.files, desc="Generating training set")):
            # Prepare 
            year = file.split(".")[0]
            image = cv2.imread(os.path.join(self.data_path, file), cv2.IMREAD_UNCHANGED)
            # Valid crop
            cv2.imwrite(os.path.join(self.val_path, f"{year}.png"), image[  max(0, self.val_center[0] - val_bbox[0]//2):min(self.spatial_size[0], self.val_center[0] + val_bbox[0]//2),
                                                                            max(0, self.val_center[1] - val_bbox[1]//2):min(self.spatial_size[1], self.val_center[1] + val_bbox[1]//2)])
            # Generating training patches' centers
            if not self.temporal or centers is None:
                centers, labels = generate_pos_neg_label_crop_centers(  self.patch_size, 
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
        val_cfg = []
        train_cfg = [[] for i in range(self.sample_size)] if self.temporal else []
         
        for branch in self.branches:
            path = os.path.join(self.train_path, branch)
            if not os.path.isdir(path):
                os.mkdir(path)
            else: 
                clear_directory(path)
        
        profiles, centers = None, None
        val_bbox = np.multiply(np.sqrt(self.val_ratio), self.spatial_size[:2]).astype(int)
        header = {'h': self.patch_size[0], 'w': self.patch_size[1]}
        for idx, file in enumerate(tqdm.tqdm(self.files, desc="Generating training set")):
            year = file.split(".")[0]
            if self.temporal:
                if os.path.isdir(img_dir):
                    subprocess.run(['rm', '-rf', img_dir])
                os.mkdir(img_dir)
            
            image = cv2.imread(os.path.join(self.data_path, file), cv2.IMREAD_UNCHANGED)
            val_img_path =  os.path.join(self.val_path, f"{year}.png")
            cv2.imwrite(val_img_path, image[  max(0, self.val_center[0] - val_bbox[0]//2):min(self.spatial_size[0], self.val_center[0] + val_bbox[0]//2),
                                                                            max(0, self.val_center[1] - val_bbox[1]//2):min(self.spatial_size[1], self.val_center[1] + val_bbox[1]//2)])
            val_cfg.append(val_img_path)
            if self.temporal:
                #(data, gnt, fg_indices, bg_indices, header, None, profiles=profiles, centers=centers)
                profiles = self.profiles if self.deterministic else None
                centers = self.centers if self.deterministic else None
                patches, self.profiles, self.centers, self.labels = transform.cut_centers_transform(image, 
                                                                            gnt=self.generator, 
                                                                            fg_indices=self.fg_indices, 
                                                                            bg_indices=self.bg_indices, 
                                                                            header=header, 
                                                                            aug=self.augment, 
                                                                            profiles=self.profiles,
                                                                            centers=self.centers,
                                                                            pos_ratio=self.label_ratio,
                                                                            num_workers=self.num_workers)
            else:
                #(data, gnt, fg_indices, bg_indices, header, None, profiles=profiles, centers=centers)
                profiles = self.profiles[idx] if self.deterministic else None
                centers = self.centers[idx] if self.deterministic else None
                patches, profiles, centers, labels = transform.cut_centers_transform(image, 
                                                                gnt=self.generator, 
                                                                fg_indices=self.fg_indices, 
                                                                bg_indices=self.bg_indices, 
                                                                header=header, 
                                                                aug=self.augment, 
                                                                profiles=profiles,
                                                                centers=centers,
                                                                pos_ratio=self.label_ratio,
                                                                num_workers=self.num_workers)
                self.profiles[idx] = profiles
                self.centers[idx] = centers
                self.labels[idx] = labels
                                            
            for index, patch in enumerate(patches):
                labels = self.labels if self.temporal else self.labels[idx]
                id = index if self.temporal else self.sample_size * idx + index
                path = os.path.join(self.train_path, self.branches[labels[index]])
                if self.temporal:
                    path = os.path.join(path, year)
                    if not os.path.isdir(path):
                        os.mkdir(path)
                train_img_path = os.path.join(path, f"{id}.png")
                cv2.imwrite(train_img_path, patch)
                if self.temporal: 
                    train_cfg[index].append(train_img_path)
                else: 
                    train_cfg.append(train_img_path)
        # Return configure for instant load
        return train_cfg, val_cfg
    
    def export_ckpt(self, path):
        output = {'profiles': self.profiles, 'centers': self.centers, 'labels': self.labels, 'temporal': self.temporal}
        np.savez(path, **output)
        print(f"Saved checkpoint to {path}")


class RiverDataset(Dataset):
    def __init__(self, data_path, temporal=False, cfg = None):
        self.data = []
        self.data_path = data_path
        self.temporal = temporal
        if not cfg:
            self.setup(data_path, temporal=temporal)
        else:
            self.data = cfg
    
    def setup(self, data_path, temporal=False):
        assert os.path.isdir(data_path)
        if not temporal:
            self.data = sorted(os.listdir(data_path))
            return

        years = sorted(os.listdir(data_path))
        if os.path.isfile(os.path.join(data_path, years[0])):
            self.data = [os.path.join(data_path, file) for file in years]
        else:
            files = sorted(os.listdir(os.path.join(data_path, years[0])))
            self.data = [[os.path.join(data_path, year, file)for year in years] for file in files]
    
    def getitem(self, index: int):
        if not isinstance(self.data[index], list):
            self.data[index] = [self.data[index]]
        return [cv2.imread(file, cv2.IMREAD_UNCHANGED) for file in self.data[index]]

    def __getitem__(self, index: int | Sequence[int]):
        if isinstance(index, int):
            return [self.getitem(index)]
        else: 
            return [self.getitem(idx) for idx in index]

    def __len__(self):
        return len(self.data)

class TransformedRiverDataset(Dataset):
    def __init__(self, dataset: Dataset, 
                        transform: Compose | None = None, 
                        hard_transform: Compose | None = None):
        self.dataset = dataset
        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        self.hard_transform = hard_transform if hard_transform is not None else self.transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # image in PIL format, landmarks in image pixel coordinates
        # data = np.transpose(self.dataset[idx], axes=[0, 3, 1, 2])
        data = self.dataset[idx]
        # print(data[0][0].shape, np.unique(data[0][0][:, :, 3]))
        inputs = []
        hard_inputs = []
        for sample in data: 
            if self.dataset.temporal:
                inputs.append(collate_batch([self.transform(image=dat[:, :, :3], mask=dat[:, :, [3]]) for dat in sample], 
                                post_act=partial(torch.stack, dim=0)))
                hard_inputs.append(collate_batch([self.hard_transform(image=dat[:, :, :3], mask=dat[:, :, [3]]) for dat in sample], 
                                post_act=partial(torch.stack, dim=0)))
            else: 
                inputs += [self.transform(image=dat[:, :, :3], mask=dat[:, :, 3]) for dat in sample]
                hard_inputs += [self.hard_transform(image=dat[:, :, :3], mask=dat[:, :, 3]) for dat in sample]
        # Will allow pixelwise augmentation only
        inputs = collate_batch(inputs, post_act=partial(torch.stack, dim=0))
        hard_inputs = collate_batch(hard_inputs, post_act=partial(torch.stack, dim=0))
        print(inputs['mask'].dtype, inputs['mask'].shape)
        labels = torch.mean(inputs['mask'], dim=tuple(range(1, len(inputs['mask'].shape))))
        inputs['label'] = torch.where(labels > 0.01, 1, 0)
        inputs['hard'] = hard_inputs['image']
        return inputs

if __name__ == "__main__":
    root = "/work/hpc/potato/airc/data"
    temporal = False
    data_path = os.path.join(root, "v2")
    suffix = "/norm" if not temporal else ""
    train_path = os.path.join(root, "dataset/v2/train") + suffix 
    val_path = os.path.join(root, "dataset/v2/val") + suffix
    augment = {'shear_range': 0.5, 'rotate_range': np.pi, 'scale_range': 0.3, 'flip_ratio': 0.5}
    pool = RiverDataPool(data_path, 
                            train_path, 
                            val_path,
                            seed=12,
                            patch_size=[256, 256],
                            sample_size=512,
                            label_ratio=0.5,
                            augment=augment,
                            temporal=temporal)
    train_cfg, val_cfg = pool.augment_crop()
    temp = RiverDataset(train_path, temporal=temporal, cfg=train_cfg)
    dataset = TransformedRiverDataset(temp, None)
    for index in range(len(dataset) - 3):
        sample = dataset[index, index + 1, index + 2]
        print(sample.keys(), sample['mask'].shape, sample['image'].shape, sample['label'])