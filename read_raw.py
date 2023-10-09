import glob
import os.path

import numpy as np
import cv2

def raw2png_0(raw_array):
    return raw_array / 16

def raw2png_1(raw_array):
    return raw_array.astype(np.uint8)

def raw2png_2(raw_array):
    max_val = np.max(raw_array)
    min_val = np.min(raw_array)
    new_raw_array = (raw_array - min_val) / max_val
    new_raw_array *= 255.0
    return new_raw_array#.astype(np.uint8)

def batch_export():
    raw_dir = r'F:\2023_songjianghe_data'
    save_dir = r'F:\2023_songjianghe_result\2'
    raw_list = glob.glob(os.path.join(raw_dir + '/*.raw'))
    for raw_file in raw_list:
        data = np.fromfile(raw_file, dtype=np.uint16)
        img_data = data.reshape(2160, 3840)
        img = raw2png_2(img_data)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(raw_file).replace('raw','png')), img)



if __name__ == '__main__':
    batch_export()