import shutil
import csv
import os
import cv2

def resize():
    img_path = r'F:\BatchSmokeClassification\x64\Release\data\512.jpg'
    img = cv2.imread(img_path)
    img_ = cv2.resize(img,(115,115))
    cv2.imwrite(r'F:\BatchSmokeClassification\x64\Release\alexnet_qua\512.jpg', img_)


def img2bin():
    img_path = r'F:\BatchSmokeClassification\x64\Release\resnet_qua\569.jpg'
    bin_path = r'F:\BatchSmokeClassification\x64\Release\resnet_qua\569.bgr'
    resize_img_path =  r'F:\BatchSmokeClassification\x64\Release\resnet_qua\569_.jpg'

    img = cv2.imread(img_path)
    resize_img = cv2.resize(img,(224,224))
    cv2.imwrite(resize_img_path, resize_img)

    img_file = img.transpose([2, 0 ,1])[None]
    img_file.tofile(bin_path)

if __name__ == '__main__':
    img2bin()