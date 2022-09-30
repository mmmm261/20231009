import time
import cv2
import glob
import os
import numpy as np

def RGB_to_label5(img):
    label = np.zeros(shape=[img.shape[0], img.shape[1]], dtype=np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if all(np.array(img[i][j]) == np.array([255, 0, 0])):  # 红⾊，建成区
                label[i][j] = 1
            elif all(np.array(img[i][j]) == np.array([0, 255, 0])):  # 绿⾊，农⽤地
                label[i][j] = 2
            elif all(np.array(img[i][j]) == np.array([0, 255, 255])):  # 天蓝⾊，林地
                label[i][j] = 3
            elif all(np.array(img[i][j]) == np.array([255, 255, 0])):  # 黄⾊，草地
                label[i][j] = 4
            elif all(np.array(img[i][j]) == np.array([0, 0, 255])):  # 蓝⾊，⽔系
                label[i][j] = 5
            else:
                label[i][j] = 0
            pass
    label = label * 51
    return label

if __name__ == '__main__':

    label_path = '/home/data/qdm_code_data/dataset/Large-scale_Classification_5classes/label_5classes'
    label_list = glob.glob(os.path.join(label_path + '/*.tif'))
    label_save_path = r'/home/data/qdm_code_data/dataset/Large-scale_Classification_5classes/label_5classes_png'
    start_time = time.time()
    for file in label_list:
        print(file)
        print(os.path.join(label_save_path, file.split('/')[-1].replace('tif', 'png')))
        img = cv2.imread(file)
        lab = RGB_to_label5(img)
        cv2.imwrite(os.path.join(label_save_path, file.split('/')[-1].replace('tif', 'png')), lab)
    end_time = time.time()
    print((end_time - start_time) / 60)


