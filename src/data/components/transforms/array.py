import cv2
import numpy as np


def point_transform(points, M, homogeneous=False, transpose=False):  
  """Extension for point-wise transformation for corners"""

  if homogeneous is False: 
    padded = np.pad(points, ((0, 0), (0, 1)), mode='constant', constant_values=(1,))
  else:
    padded = points
  
  if transpose is True:
    padded[:, :2] = np.roll(padded[:, :2], 1, axis=1)
  transformed = np.vstack([M @ p for p in padded])
  
  if transpose is True:
    transformed[:, :2] = np.roll(transformed[:, :2], -1, axis=1)
  
  if homogeneous is True:
    return transformed
  else:
    return transformed[:, :2]

def transform_compose(h, w, shear_x, shear_y,  scale_x, scale_y, angle):
  """ Shape transform wrapper by fore-matmul all transformation 
  """

  # shear transformation
  shear_kernel = np.array([ [scale_x, shear_x * scale_y, 0],
                              [shear_y * scale_x, scale_y, 0],
                              [0, 0, 1]])
  
  # skew transformation
  rotate = cv2.getRotationMatrix2D((int((w * scale_x + h * shear_x * scale_y) / 2), int((h * scale_y + w * shear_y * scale_x) / 2)), angle=-angle, scale=1.)
  rotate = np.pad(skew, ((0, 1), (0, 0)), mode='constant', constant_values=(0,))
  rotate[2, 2] = 1
  
  transform =  shear_kernel @ rotate
  inv_transform = np.linalg.inv(transform)

  return transform, inv_transform

def affine_cut(img, pt, h, w, augment):
  transform, inv_transform = transform_compose(h, w,**augment)
  # padding
  corner = np.zeros((4, 2))
  corner[1:3, 1] += h
  corner[2:4, 0] += w
  # Inverse transform to get there
  crop_corner = point_transform(corner, inv_transform, homogeneous=False) 
  translation = np.max(crop_corner, axis=0) - np.min(crop_corner, axis=0)
  crop_corner = crop_corner - translation / 2 + pt
  

