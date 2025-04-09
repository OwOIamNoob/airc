import sam2
print(sam2.__file__)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor as SAM
import cv2
import numpy as np
# import torch
from PIL import  Image
import matplotlib.pyplot as plt
import scipy
import skimage

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.backends.cudnn.enabled)
# print(torch.backends.cudnn.version())
# Reading image
image = cv2.imread("data/v2/2015.png", cv2.IMREAD_UNCHANGED)
mask = image[:, :, 3]
image = image[:, :, :3]

# SAM needs RGB format so...
img = np.asarray(image[:512, :512, ::-1].copy())

print("Initiating model")
checkpoint = "/work/hpc/potato/sam/sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
sam = build_sam2(model_cfg, checkpoint)
# sam.to("cuda:2")
model = SAM(sam)

print("Encoding")
print(type(img))
# with torch.cuda.amp.autocast(enabled=False):
model.set_image(img)
