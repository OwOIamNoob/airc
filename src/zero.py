from typing import Any, Dict, List, Optional, Tuple
import hydra
from  omegaconf import DictConfig
import pandas as pd
import os
import sys
from copy import copy
import shutil
import logging
logging.disable(logging.CRITICAL + 1) 
os.environ['HYDRA_FULL_ERROR'] = '1'
print(sys.getrecursionlimit())
sys.setrecursionlimit(1000000000)

import tqdm
import json
import rootutils
from pathlib import Path
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.zero_shot import *

from collections import defaultdict
from functools import  partial
from queue import PriorityQueue, Queue
import warnings
import heapq
from heapq import heappush, heappop


import cv2
import numpy as np
from PIL import  Image
import scipy
import skimage

import segmentationmetrics as segmetrics

import torch
print(torch.cuda.is_available())
import sam2
from sam2.build_sam import build_sam2, build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor as SAM

from hydra.core.global_hydra import GlobalHydra

import matplotlib.pyplot as plt
from src.zero_shot.sam import SamInferer
from src.zero_shot.utils import save_gray, save_grey, sigmoid
import src.zero_shot.utils as utils

if GlobalHydra.instance().is_initialized():
    print("GlobalHydra is already initialized. Reinitializing")
else:
    print("GlobalHydra is not initialized.")

print("IMPORTED")

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def prepare(out_dir):
    # os.system(f"rm -rf {out_dir}")
    # if os.path.isdir(out_dir):
    #     shutil.rmtree(out_dir)
    (out_dir / "strong").mkdir(parents=True, exist_ok=True)
    (out_dir / "weak").mkdir(parents=True, exist_ok=True)
    (out_dir / "logit").mkdir(parents=True, exist_ok=True)
    (out_dir / "output").mkdir(parents=True, exist_ok=True)
    (out_dir / "image").mkdir(parents=True, exist_ok=True)
    (out_dir / "thin").mkdir(parents=True, exist_ok=True)
    (out_dir / "skeleton").mkdir(parents=True, exist_ok=True)
    (out_dir / "ensembled_output").mkdir(parents=True, exist_ok=True)
    (out_dir / "confidence").mkdir(parents=True, exist_ok=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="sam.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    data_dir = Path(cfg.paths.data_dir)
    # data_dir.mkdir(parents=True, exist_ok=True)
    # print(data_dir)
    param = hydra.utils.instantiate(cfg.model)
    marker = cv2.imread(r"E:\river_seg\airc\data\negative_region.png")[..., -1]
    # print(np.unique(marker, return_counts=True), marker.shape)
    marker = (marker < 250)
    assert os.path.isdir(data_dir), "Data dir must be a dir"
    for file in sorted(os.listdir(data_dir)):
        # print(file)
        year = file.split(".")[0]
        image_path = os.path.join(data_dir, file)
        root_dir = Path(cfg.paths.output_dir)
        # plt.imshow(skimage.morphology.skeletonize(output['possible']))
        # print("Inferencing on SAMPLE", image_path)
        # Prepare for ablation study
        for beta in np.arange(0.5, 0.9, step=0.05):
            # print("Inferencing on beta", beta)
            param.beta = beta
            trial = 0
            for trial in range(5):
                param.reset()
                param.marker = marker.copy()
                out_dir = root_dir / f"{trial:02d}" / f"{beta:.2f}" / f"{year}"
                prepare(out_dir)
                eval_res = {'dice': [], 'iou': [], 'acc': [], 'recall': [], 'f1': []}
                iterations = 0
                param.read(image_path)
                param.add_queue({'pt': copy(cfg.prompt)}, isroot=True)
                param.neg = np.array([[275, 1100], [365, 1360], [900, 2050], [910, 2050], [5964, 3275]])
                pbar = tqdm.tqdm(range(200), desc="Inference process")
                for iter in pbar:
                    pbar.refresh()
                    pbar.set_description_str(f"{year}_{beta:.2f}_{trial:02d}")
                    filename = f"iter_{iter}"
                    if len(param.queue) == 0:
                        break
                    iterations += 1
                    # print(f"Iteration {iterations}")
                    output = param.iter(debug=True)
                    if output['ret'] is True:           
                        if eval_res is None: 
                            eval_res = dict(zip(output['metrics'].keys(), [[]] * len(output['metrics'])))
                        # print(output['metrics'])
                        # print(output['infer']['logit'].min())
                        save_gray(f"{out_dir}/logit/{filename}.jpg", sigmoid(np.concatenate([sigmoid(output['infer']['logit']), output['prob_map']], axis=0)), 'viridis', output_size=cfg.log_size)
                        save_gray(f"{out_dir}/thin/{filename}.jpg", output['thin'], 'viridis', nonzero=False, output_size=cfg.log_size)
                        save_gray(f"{out_dir}/weak/{filename}.jpg", output['possible'], 'viridis', nonzero=False, output_size=cfg.log_size)
                        save_gray(f"{out_dir}/strong/{filename}.jpg", output['beta'], 'viridis', nonzero=False, output_size=cfg.log_size)
                        # save_gray(f"{out_dir}/output/{filename}.jpg", output['prob_map'], 'viridis', output_size=cfg.log_size)
                        # save_gray(f"{out_dir}/confidence/{filename}.jpg", np.concatenateoutput['prob_map'], 'viridis', output_size=cfg.log_size)
                        save_gray(f"{out_dir}/skeleton/{filename}.jpg", output['skeleton'], 'viridis', invert=True, output_size=cfg.log_size)
                        image = output['infer']['input'].copy().astype('float') / 255
                        input_mask = scipy.ndimage.morphological_gradient(output['infer']['inp_mask'], size=3)
                        image = image * (1 - input_mask[..., None]) + input_mask[..., None] * np.array([1, 1, 0], dtype=np.uint8)[None, None, :]
                        annotation = output['infer']['pts']
                        a_label = output['infer']['label'][:, None]
                        color = [0, 1, 0, 0.5] * a_label + [1, 0, 0, 0.5] * (1 - a_label)
                        for pt, c in zip(annotation, color):
                            cv2.circle(image, pt, 5, c, -1)
                        cv2.imwrite(f"{out_dir}/image/{filename}.jpg", (image * 255)[..., ::-1].astype(np.uint8))
                        for key in output['metrics'].keys():
                                eval_res[key].append(float(output['metrics'][key]))
                    else:
                        # print("Hehe")
                        pass
                # print(param.roi)
                # with open(f"{out_dir}/micro_{'_'.join([str(item) for item in param.roi[2:]])}.txt", "w") as file:
                    # file.write(f"Inferences: {iter}\n")
                    # for key in eval_res.keys():
                    #     mean = np.mean(eval_res[key])
                    #     std = np.std(eval_res[key])
                    #     file.write(f"{key} metrics: {mean} +- {std}\n")
                df = pd.DataFrame(eval_res)
                df.to_csv(f"{out_dir}/micro_{'_'.join([str(item) for item in param.roi[2:]])}.csv")
                src, dst = [0, 0], param.box
                # src[0] = max(400, src[0])
                # weight = scipy.ndimage.gaussian_filter(param.weight, sigma=2.)
                logit = param.logits[src[0]:dst[0], src[1]:dst[1]] / param.weight
                if not param.post_act:
                    prob = sigmoid(logit)
                else:
                    prob = logit
                # beta = param.beta / param.weight
                # logit = logit ** 2
                label = param.label[src[0]:dst[0], src[1]:dst[1]]
                save_grey(f"{out_dir}/ensembled_output/logit.jpg", prob)
                save_grey(f"{out_dir}/ensembled_output/mask.jpg", param.b_mask)
                quantized = np.interp(prob ** 1.5, (0, 1), (0, 255)).astype(int).astype(np.uint8)
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(251, 251))
                quantized = clahe.apply(quantized)
                # print(quantized.shape, quantized.dtype)
                save_grey(f"{out_dir}/ensembled_output/quantized.jpg", quantized)
                b_mask = (quantized >= 150)
                b1 = cv2.adaptiveThreshold(quantized, 
                                                    1, 
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 
                                                    91, 
                                                    -15) * (quantized > 50) 
                b2 = cv2.adaptiveThreshold(quantized, 
                                                    1, 
                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 
                                                    121, 
                                                    -15) * (quantized > 80)
                mask = b_mask + (1 - b_mask) * np.maximum(b1, b2)
                mask = mask + (1 - mask) * param.b_mask[src[0]:dst[0], src[1]:dst[1]] * (quantized >= 10)
                # mask *= param.marker
                mask = utils.prune(mask, min_size=100)
                output = np.zeros(mask.shape, np.uint8)
                _, label_im = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
                labs, counts = np.unique(label_im, return_counts=True)
                for lab, count in zip(labs[1:], counts[1:]):
                    im = label_im == lab
                    if param.b_mask[src[0]:dst[0], src[1]:dst[1]][im].sum() <= 5:
                        continue 
                    else: 
                        output = np.maximum(output, im)
                mask = output
                mask = utils.prune(mask, min_size=500)
                mask_grad = scipy.ndimage.morphological_gradient(mask, size=3)
                label_grad = scipy.ndimage.morphological_gradient(label, size=3)
                image = param.image.copy()
                res = segmetrics.SegmentationMetrics(mask,label,(1, 1))
                image[...,1][mask_grad > 0] = 255
                image[...,0][label_grad > 0] = 255
                with open(f"{out_dir}/macro_{beta}.txt", "w") as file:
                    file.write(str(res.get_df()))
                cv2.imwrite(f"{out_dir}/ensembled_output/annotated.jpg", image[..., ::-1])
                pbar.write(f"{year}_{beta:.2f}_{trial:02d}:\n" + res.get_df().to_string())
    return True

if __name__ == "__main__":
    print("Proceed")
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    main()
