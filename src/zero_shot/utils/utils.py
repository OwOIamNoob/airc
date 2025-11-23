import numpy as np
import cv2
import matplotlib as mpl

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_bbox(bound, pt, patch_size):
    x = min(bound[1] - patch_size[1], max(0, pt[1] - patch_size[1] // 2))
    y = min(bound[0] - patch_size[0], max(0, pt[0] - patch_size[0] // 2))
    return x, y, patch_size[0], patch_size[1]

def save_gray(filename, image, cmap, invert=False, nonzero=True, output_size=None, verbose = False):
    order = 1 if not invert else -1
    if nonzero:
        normed = image.copy()
        masked = image[image != 0]
        normed[normed != 0] = np.interp(masked, (masked.min(), masked.max()), (0, 1)[::order])
    else:
        normed = np.interp(image, (image.min(), image.max()), (0, 1)[::order])
    rgb_image = mpl.colormaps[cmap](normed)
    bgr_image = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR) 
    if output_size is not None:
        bgr_image = cv2.resize(bgr_image, output_size)
    ret = cv2.imwrite(filename, bgr_image)
    if verbose:
        print(ret)

def save_grey(filename, image, verbose=False):
    normed = np.interp(image, (image.min(), image.max()), (0, 255)).astype(int).astype(np.uint8)
    ret = cv2.imwrite(filename, normed[..., None])
    if verbose:
        print(ret)