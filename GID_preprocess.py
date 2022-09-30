import os
import numpy as np
import cv2
from osgeo import gdal
from PIL import Image
import glob
import csv
def readTif(fileName, file_type = np.uint8):
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "鎺╄啘澶辫触锛屾枃浠舵棤娉曟墦寮€")

    im_width = dataset.RasterXSize  # 鏍呮牸鐭╅樀鐨勫垪鏁?
    im_height = dataset.RasterYSize  # 鏍呮牸鐭╅樀鐨勮鏁?

    im_bands = dataset.RasterCount  # 娉㈡鏁?

    im_geotrans = dataset.GetGeoTransform()  # 鑾峰彇浠垮皠鐭╅樀淇℃伅
    im_proj = dataset.GetProjection()  # 鑾峰彇鎶曞奖淇℃伅

    img = np.zeros((im_height, im_width, im_bands), dtype=file_type)

    for i in range(im_bands):
        band = dataset.GetRasterBand(i + 1)
        band_data = band.ReadAsArray(0, 0, im_width, im_height).astype(file_type)
        img[:, :, i] = band_data
        if file_type == np.uint16:
            img[:, :, i] = cv2.convertScaleAbs(band_data, alpha=(255.0 / 65535.0))
    img = img[:,:,::-1]
    return img

def RGB_to_labe15l(img):
    label = np.zeros(shape=[img.shape[0], img.shape[1]], dtype=np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if all(np.array(img[i][j]) == np.array([200, 0, 0])):  # industrial land 1
                label[i][j] = 1
            elif all(np.array(img[i][j]) == np.array([250, 0, 150])):  # urban residential  2
                label[i][j] = 2
            elif all(np.array(img[i][j]) == np.array([200, 150, 150])):  # rural residential 3
                label[i][j] = 3
            elif all(np.array(img[i][j]) == np.array([250, 150, 150])):  # traffic land 4
                label[i][j] = 4
            elif all(np.array(img[i][j]) == np.array([0, 200, 0])):  # paddy field 5
                label[i][j] = 5
            elif all(np.array(img[i][j]) == np.array([150, 250, 0])):  # irrigated land
                label[i][j] = 6
            elif all(np.array(img[i][j]) == np.array([150, 200, 150])):  # dry cropland
                label[i][j] = 7
            elif all(np.array(img[i][j]) == np.array([200, 0, 200])):  # garden plot
                label[i][j] = 8
            elif all(np.array(img[i][j]) == np.array([150, 0, 250])):  # arbor woodland
                label[i][j] = 9
            elif all(np.array(img[i][j]) == np.array([150, 150, 250])):  # shrub land
                label[i][j] = 10
            elif all(np.array(img[i][j]) == np.array([250, 200, 0])):  # natural grassland
                label[i][j] = 11
            elif all(np.array(img[i][j]) == np.array([200, 200, 0])):  # artificial grassland
                label[i][j] = 12
            elif all(np.array(img[i][j]) == np.array([0, 0, 200])):  # river
                label[i][j] = 13
            elif all(np.array(img[i][j]) == np.array([0, 150, 200])):  # lake
                label[i][j] = 14
            elif all(np.array(img[i][j]) == np.array([0, 200, 250])):  # pound
                label[i][j] = 15
            else:
                label[i][j] = 0
            pass
    return label

def RGB_to_label5(img):
    label = np.zeros(shape=[img.shape[0], img.shape[1]], dtype=np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if all(np.array(img[i][j]) == np.array([255, 0, 0])):  # 绾⑩緤锛屽缓鎴愬尯
                label[i][j] = 1
            elif all(np.array(img[i][j]) == np.array([0, 255, 0])):  # 缁库緤锛屽啘饨ゅ湴
                label[i][j] = 2
            elif all(np.array(img[i][j]) == np.array([0, 255, 255])):  # 澶╄摑饩婏紝鏋楀湴
                label[i][j] = 3
            elif all(np.array(img[i][j]) == np.array([255, 255, 0])):  # 榛勨緤锛岃崏鍦?
                label[i][j] = 4
            elif all(np.array(img[i][j]) == np.array([0, 0, 255])):  # 钃濃緤锛屸綌绯?
                label[i][j] = 5
            else:
                label[i][j] = 0
            pass
    label = label * 51
    return label

def get_img_mask_list():
    origal_train_list_path = r'E:\pycode\Remote_sensing_segmentation\data_list\origal_train_list.txt'
    origal_test_list_path = r'E:\pycode\Remote_sensing_segmentation\data_list\origal_test_list.txt'
    tif_block_path = r'E:\pycode\datasets\GID\Large-scale Classification_5classes\image_RGB_block'
    train_list_path = r'E:\pycode\Remote_sensing_segmentation\data_list\train_list.txt'
    test_list_path = r'E:\pycode\Remote_sensing_segmentation\data_list\test_list.txt'

    with open(origal_train_list_path, 'r') as f:
        train_list = f.readlines()
    print(train_list)

    with open(train_list_path, 'a') as ff:
        for train_name in train_list:
            train_name = train_name.strip('\n')
            print(train_name)
            for tif_file in os.listdir(tif_block_path):
                if tif_file.startswith(train_name):
                    ff.write(tif_file+'\n')

    with open(origal_test_list_path, 'r') as fff:
        test_list = fff.readlines()

    with open(test_list_path, 'a') as ffff:
        for test_name in test_list:
            test_name = test_name.strip('\n')
            for tif_file in os.listdir(tif_block_path):
                if tif_file.startswith(test_name):
                    ffff.write(tif_file+'\n')

def calculate_class_distribution(label_path):
    csv_path = os.path.join(os.path.dirname(label_path), 'class_distribution.csv')
    label_list = glob.glob(os.path.join(label_path + '\\*.png'))
    all_cal_list = []
    for label_file in label_list:
        one_cal_list = []
        label = cv2.imread(label_file, 0)
        label = label / 51
        one_cal_list.append(label_file.split('/')[-1].split('.')[0])
        one_cal_list.append(np.sum(label == 1)/label.size)
        one_cal_list.append(np.sum(label == 2)/label.size)
        one_cal_list.append(np.sum(label == 3)/label.size)
        one_cal_list.append(np.sum(label == 4)/label.size)
        one_cal_list.append(np.sum(label == 5)/label.size)
        one_cal_list.append(np.sum(label == 0)/label.size)

        all_cal_list.append(one_cal_list)

    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "build_up", "farmland", "forest", "meadow", "water", "others"])
        for r in all_cal_list:
            writer.writerow(r)

class GID_process():
    def __init__(self, tif_path, infer_size = 400):

        self.tif_path = tif_path
        self.tif_dir = os.path.dirname(tif_path)
        self.tif_name = os.path.basename(tif_path).split('.')[0]
        self.infer_size = infer_size
        self.padding_size = int(0.5 * infer_size)
        self.block_save_path = os.path.join(self.tif_path.split('.')[0] + '_block')
        #self.block_save_path = os.path.join(os.path.dirname(os.path.dirname(tif_path)), 'label_5classes_png_block')

        self.padding_block_save_path = os.path.join(self.tif_path.split('.')[0] + '_padding_block')
        #self.merge_save_path = os.path.join(self.tif_path.split('.')[0] + '_result')
        self.merge_save_path = r'E:\pycode\Remote_sensing_segmentation\run\test_result'
        try:
            h, w, c = cv2.imread(self.tif_path, -1).shape
            self.origal_h, self.origal_w, self.origal_c = h, w, c
        except:
            h, w= cv2.imread(self.tif_path, -1).shape
            self.origal_h, self.origal_w, self.origal_c = h, w, 1

        self.new_h = h if h % self.infer_size == 0 else (int(h / self.infer_size) + 1) * self.infer_size
        self.new_w = w if w % self.infer_size == 0 else (int(w / self.infer_size) + 1) * self.infer_size

    def cut_tif(self, cut_padding = False):
        if not os.path.exists(self.block_save_path):
            os.mkdir(self.block_save_path)

        self.tif_img = readTif(self.tif_path) if self.tif_path.endswith('tif') else cv2.imread(self.tif_path, 0)
        mask_tif = np.zeros((self.new_h, self.new_w, self.origal_c)) if self.origal_c > 1 else np.zeros((self.new_h, self.new_w))
        try:
            mask_tif[:self.origal_h, :self.origal_w, :] = self.tif_img
        except:
            mask_tif[:self.origal_h, :self.origal_w] = self.tif_img

        for i in range(int(self.new_h / self.infer_size)):
            for j in range(int(self.new_w / self.infer_size)):
                img_block = mask_tif[self.infer_size * i: self.infer_size * i + self.infer_size, self.infer_size * j: self.infer_size * j + self.infer_size, :] \
                if self.origal_c > 1 else mask_tif[self.infer_size * i: self.infer_size * i + self.infer_size, self.infer_size * j: self.infer_size * j + self.infer_size]

                img_block_name = os.path.join(self.block_save_path,  "{}_{}_{}.png".format(os.path.basename(self.tif_path).split('.')[0], i, j))
                cv2.imwrite(img_block_name, img_block)

        if cut_padding:
            if not os.path.exists(self.padding_block_save_path):
                os.mkdir(self.padding_block_save_path)

            mask_tif = np.zeros((self.new_h + self.infer_size, self.new_w + self.infer_size, self.origal_c))
            mask_tif[self.padding_size:self.origal_h + self.padding_size,
            self.padding_size:self.origal_w + self.padding_size, :] = self.tif_img

            for i in range(int((self.new_h + self.infer_size) / self.infer_size)):
                for j in range(int((self.new_w + self.infer_size) / self.infer_size)):
                    img_block = mask_tif[self.infer_size * i: self.infer_size * i + self.infer_size,
                                self.infer_size * j: self.infer_size * j + self.infer_size, :]
                    img_block_name = os.path.join(self.padding_block_save_path,
                                                  "{}_{}_{}.png".format(os.path.basename(self.tif_path).split('.')[0],
                                                                        i, j))
                    cv2.imwrite(img_block_name, img_block)


    def merge_tif(self, pred_path, padding_path = ''):
        if not os.path.exists(self.merge_save_path):
            os.mkdir(self.merge_save_path)

        pred_block_list = [i.split('.')[0] for i in os.listdir(pred_path) if i.startswith(self.tif_name)]

        mask_result = np.zeros((self.new_h, self.new_w))

        for block_file in pred_block_list:
            idx_i = int(block_file.split('_')[-2])
            idx_j = int(block_file.split('_')[-1])
            print("idx_i:", idx_i, "idx_j:", idx_j)
            print(mask_result[idx_i * self.infer_size: idx_i * self.infer_size + self.infer_size, idx_j * self.infer_size: idx_j * self.infer_size + self.infer_size].shape)
            print(idx_j * self.infer_size, idx_j * self.infer_size + self.infer_size)
            mask_result[idx_i * self.infer_size: idx_i * self.infer_size + self.infer_size, idx_j * self.infer_size: idx_j * self.infer_size + self.infer_size] = cv2.imread(
                os.path.join(pred_path, "{}.png".format(block_file)), -1)

        mask_result = mask_result[:self.origal_h, :self.origal_w]

        if padding_path:
            padding_pred_block_list = [i.split('.')[0] for i in os.listdir(padding_path)]
            padding_mask_result = np.zeros((self.new_h + self.infer_size, self.new_w + self.infer_size))

            for padding_block_file in padding_pred_block_list:
                padding_idx_i = int(padding_block_file.split('_')[-2])
                padding_idx_j = int(padding_block_file.split('_')[-1])
                padding_mask_result[padding_idx_i * self.infer_size: padding_idx_i * self.infer_size + self.infer_size,
                padding_idx_j * self.infer_size: padding_idx_j * self.infer_size + self.infer_size] = cv2.imread(
                    os.path.join(padding_path, "{}.png".format(padding_block_file)), -1)

            padding_mask_result = padding_mask_result[self.padding_size:self.origal_h + self.padding_size,
            self.padding_size:self.origal_w + self.padding_size]

            mask_result = mask_result / 255
            padding_mask_result = padding_mask_result / 255
            mask_result += padding_mask_result

            #鍙栦氦闆?
            #mask_result[mask_result < 2] = 0
            #mask_result[mask_result == 2] = 1

            #鍙栧苟闆?
            mask_result[mask_result > 1] = 1

            mask_result = mask_result * 255


        mask_name = os.path.join(self.merge_save_path, '{}_result.png'.format(self.tif_name)) if not padding_path else \
        os.path.join(self.merge_save_path, '{}_padding_result.png'.format(self.tif_name))

        cv2.imwrite(mask_name, mask_result)

    def save_png(self, png_save_path):
        tif_img_ = readTif(self.tif_path)
        cv2.imwrite(png_save_path, tif_img_)

def test_merge():
    tif_list = glob.glob(
        os.path.join(r'E:\pycode\datasets\GID_dataset\Large-scale Classification_5classes\image_RGB' + '\\*.png'))
    pred_path_ = r'E:\pycode\Remote_sensing_segmentation\run\test\resnet101_pretrained_DANet_20220924_epoch15_test_list'
    for tif_file in tif_list:
        gid = GID_process(tif_file, 480)
        gid.merge_save_path = r'E:\pycode\Remote_sensing_segmentation\run\resnet101_pretrained_DANet_20220924_epoch15'
        gid.merge_tif(pred_path_)

if __name__ == '__main__':
    test_merge()



















    # for file in file_list:
    #     img_path = os.path.join(r'Z:\qdm\GID_dataset\Large-scale Classification_5classes\image_RGB', "{}.tif".format(file))
    #     img_save_path = os.path.join(r'E:\pycode\datasets\GID_dataset\Large-scale Classification_5classes\image_RGB', "{}.png".format(file))
    #     print(img_save_path)
    #     img_array = readTif(img_path)
    #     cv2.imwrite(img_save_path, img_array)

    # for file in file_list:
    #     lab_path = os.path.join(r'Z:\qdm\GID_dataset\Large-scale Classification_5classes\label_5classes', "{}_label.tif".format(file))
    #     lab_save_path = os.path.join(r'E:\pycode\datasets\GID_dataset\Large-scale Classification_5classes\label_5classes', "{}.png".format(file))
    #     print(lab_save_path)
    #     lab_array = readTif(lab_path)
    #     cv2.imwrite(lab_save_path, lab_array)
