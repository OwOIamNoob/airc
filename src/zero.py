from typing import Any, Dict, List, Optional, Tuple
import hydra
from  omegaconf import DictConfig

import os
import sys
from copy import copy
print(sys.getrecursionlimit())
sys.setrecursionlimit(1000000000)

import tqdm
import json
import rootutils
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
    os.system(f"rm -rf {out_dir}")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        os.mkdir(f"{out_dir}/strong")
        os.mkdir(f"{out_dir}/weak")
        os.mkdir(f"{out_dir}/logit")
        os.mkdir(f"{out_dir}/output")
        os.mkdir(f"{out_dir}/image")
        os.mkdir(f"{out_dir}/thin")
        os.mkdir(f"{out_dir}/skeleton")
        os.mkdir(f"{out_dir}/ensembled_output")
        os.mkdir(f"{out_dir}/confidence")

@hydra.main(version_base="1.3", config_path="../configs", config_name="sam.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    data_dir = cfg.paths.data_dir
    print(data_dir)
    param = hydra.utils.instantiate(cfg.model)
    assert os.path.isdir(data_dir), "Data dir must be a dir"
    for file in sorted(os.listdir(data_dir)):
        print(file)
        year = file.split(".")[0]
        image_path = os.path.join(data_dir, file)
        out_dir = os.path.join(cfg.paths.output_dir, year)
        prepare(out_dir)
        # plt.imshow(skimage.morphology.skeletonize(output['possible']))
        print("Inferencing on SAMPLE", image_path)
        # Prepare for ablation study
        for beta in [0.6]:
            print("Inferencing on beta", beta)
            param.beta = beta
            for trial in range(5):
                param.reset()
                eval_res = {'dice': [], 'iou': [], 'acc': [], 'recall': [], 'f1': []}
                iterations = 0
                param.read(image_path)
                param.add_queue(copy(cfg.prompt), isroot=True)
                for iter in tqdm.tqdm(range(200), desc="Inference process"):
                    if len(param.queue) == 0:
                        break
                    iterations += 1
                    # print(f"Iteration {iterations}")
                    output = param.iter(debug=True)
                    print(param.roi)
                    if output['ret'] is True:           
                        if eval_res is None: 
                            eval_res = dict(zip(output['metrics'].keys(), [[]] * len(output['metrics'])))
                        print(output['metrics'])
                        for key in output['metrics'].keys():
                                eval_res[key].append(float(output['metrics'][key]))
                    else:
                        print("Hehe")
                print(param.roi)
                if param.roi[2] > 6000 and param.roi[3] > 3000:
                    print(f"Attempt {trial} succeeded, proceed")
                    break
                else: 
                    print("Retrying")
                    
            with open(f"{out_dir}/micro_{beta}.txt", "w") as file:
                file.write(f"Inferences: {iter}\n")
                for key in eval_res.keys():
                    mean = np.mean(eval_res[key])
                    std = np.std(eval_res[key])
                    file.write(f"{key} metrics: {mean} +- {std}")
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
            save_grey(f"{out_dir}/ensembled_output/{year}_logit_v2.jpg", prob)
            save_grey(f"{out_dir}/ensembled_output/{year}_mask_v2.jpg", param.b_mask)
            quantized = np.interp(prob ** 1.5, (0, 1), (0, 255)).astype(int).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(251, 251))
            quantized = clahe.apply(quantized)
            # save_grey(f"{out_dir}/ensembled_output/{year}_beta_v2.jpg", beta)
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
            label[:355] = 0
            mask[:355] = 0
            res = segmetrics.SegmentationMetrics(mask,label,(1, 1))
            with open(f"{out_dir}/macro_{beta}.txt", "w") as file:
                file.write(str(res.get_df()))
            print(res.get_df())
            print("Continue !!!!")
    return True

if __name__ == "__main__":
    print("Proceed")
    from hydra.core.global_hydra import GlobalHydra
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    main()
