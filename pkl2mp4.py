import glob
import os.path
import pickle
from PIL import Image
import io
import numpy as np
import cv2

def pkl2mp4(pkl_path, mp4_path):
    f = open(pkl_path, 'rb')
    pkl_img_data = pickle.loads(f.read())[2]
    f.close()

    video_write = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), fps = 5, frameSize = (256, 256))

    for i, io_data in enumerate(pkl_img_data):
        image = Image.open(io.BytesIO(io_data))
        array = np.array(image)[:,:,::-1]
        video_write.write(array)
    video_write.release()

if __name__ == '__main__':

    pkl_list = glob.glob(os.path.join(r'C:\Users\1\Desktop\test_data' + '\\*.pkl'))
    for k, pkl_path in enumerate(pkl_list):
        mp4_path = pkl_path.replace('pkl','mp4')
        pkl2mp4(pkl_path, mp4_path)

