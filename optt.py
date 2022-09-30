import cv2
import numpy as np
import random
import math

im = cv2.imread(r'E:\pycode\datasets\cgwx\train\image\000001.tif',-1)
im = np.load(r'E:\pycode\datasets\cgwx\train\bgra_cut_npys\000001.npy')
print(im.shape)

border=(0, 0)
perspective=0.0
degrees=10
scale=.1
shear=10
translate=.1

height = im.shape[0] + border[0] * 2  # shape(h,w,c)
width = im.shape[1] + border[1] * 2

# Center
C = np.eye(3)
C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

# Perspective
P = np.eye(3)
P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

# Rotation and Scale
R = np.eye(3)
a = random.uniform(-degrees, degrees)
# a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
s = random.uniform(1 - scale, 1 + scale)
# s = 2 ** random.uniform(-scale, scale)
R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

# Shear
S = np.eye(3)
S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

# Translation
T = np.eye(3)
T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
print("M:",M.shape)
if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
    if perspective:
        print(1)
        im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
    else:  # affine
        print(2)
        im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))