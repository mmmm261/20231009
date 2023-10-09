import glob
import os
import shutil
import numpy as np
import csv
import cv2
import io

from PIL import Image
def dd_14():
    import ffmpy  # 导入
    for file in os.listdir(r'F:\wchat_voice_tmp\autovc-master\data\8_liu'):
        fileOldPath = os.path.join(r'F:\wchat_voice_tmp\autovc-master\data\8_liu', file)
        fileNewPath = os.path.join(r'F:\wchat_voice_tmp\autovc-master\data\8_liu_wav', file.replace('mp3', 'wav'))

        ff = ffmpy.FFmpeg(
            inputs={fileOldPath: None},
            outputs={fileNewPath: None}
        )
        ff.run()

def dd_15():
    dir = r'F:\wchat_voice_tmp\voice_file\huawei_voice'
    for path in os.listdir(dir):
        for path_2 in os.listdir(path):
            amr_dir = os.path.join(dir, path, path_2)
            print(amr_dir)
            break

def dd_16():
    img = cv2.imread(r'C:\Users\1\Desktop\tmp\tmp_20230412_data\hw(1).jpg')
    print(img.shape)
    new_img = cv2.resize(img, (1920, 1080))
    cv2.imwrite(r'C:\Users\1\Desktop\tmp\tmp_20230412_data\hw_1080.jpg', new_img)

def dd_17():
    img = cv2.imread(r'C:\Users\1\Desktop\wordpress_demo\2.jpg')
    mask = cv2.imread(r'C:\Users\1\Desktop\tmp\segment_anything_result\other\2\2_23_iou0.9884507656097412_score0.968987226486206.png')

    block = img * (mask / 255)
    cv2.imwrite(r'C:\Users\1\Desktop\wordpress_demo\2_mask.jpg', block)

def dd_18():
    txt_file = r'E:\pycode\datasets\DanZhenYanHuo_mini\val\labels\32.txt'
    with open(txt_file, 'r') as f:
        files = f.readlines()
    point_list = []
    for lines in files:
        line = [float(i) for i in lines.split()[1:]]
        print(line)

def dd_19():
    points = []
    for i in range(3):
        point = np.array([10,10,20,20])
        points.append(point)
    print(points)
    b = np.vstack([i for i in points])
    print(b)

def dd_20():
    csv_path = r'E:\cppcode\weight_backup_standard\yolov5x_pretrained_SGD_scratch-high_image-weights_20230309\results\best.csv'
    with open(csv_path, "r") as f:
        f.readline()
        csv_file = f.readlines()
    csv_file = csv_file[:-1]
    print(len(csv_file))
    wb_file_list = []
    lb_file_list = []
    dd_file_list = []
    for file in csv_file:
        file_name, lab_num, pre_num, LB_num, WB_num = file.split(',')
        if int(WB_num) > 0 and int(LB_num) == 0:
            wb_file_list.append(file_name)
        if int(LB_num) > 0 and int(WB_num) == 0:
            lb_file_list.append(file_name)
        if int(WB_num) == 0 and int(LB_num) == 0:
            dd_file_list.append(file_name)
    print(len(wb_file_list))
    print(len(lb_file_list))
    print(len(dd_file_list))
    for wb_file in wb_file_list:
        wb_name = os.path.basename(wb_file).replace('jpg', 'txt')
        shutil.copy(os.path.join(r'E:\pycode\datasets\tmp_SAM\wb_file\results\labels',wb_name), os.path.join(r'E:\pycode\datasets\tmp_SAM\wb_file\labels',wb_name))
    '''
    for lb_file in lb_file_list:
        lb_name = os.path.basename(lb_file).replace('jpg', 'txt')
        shutil.copy(os.path.join(r'E:\pycode\datasets\DZYH_test_dataset\yancut20220802_test_smoke\txt',lb_name), os.path.join(r'E:\pycode\datasets\tmp\lb_file',lb_name))
    for dd_file in dd_file_list[:200]:
        dd_name = os.path.basename(dd_file).replace('jpg', 'txt')
        shutil.copy(os.path.join(r'E:\pycode\datasets\DZYH_test_dataset\yancut20220802_test_smoke\txt',dd_name), os.path.join(r'E:\pycode\datasets\tmp\dd_file',dd_name))
    '''

def dd_21():
    from osgeo import gdal
    import cv2
    gid_dir = r'Z:\qdm\GID_dataset\Large-scale Classification_5classes\image_RGB'
    gid_file_list = glob.glob(os.path.join(gid_dir + "\\*.tif"))
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    gid_rgb_value = [0, 0, 0]

    for j, fileName in enumerate(gid_file_list):
        dataset = gdal.Open(fileName)
        im_bands = dataset.RasterCount
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        file_rgb_value = [0,0,0]
        for i in range(im_bands):
            band = dataset.GetRasterBand(i + 1)
            band_data = band.ReadAsArray(0, 0, im_width, im_height).astype(np.uint8)
            mean_value = np.mean(band_data)
            file_rgb_value[i] += mean_value
            gid_rgb_value[i] += mean_value
        print("file index:", j , "file_rgb_value:", file_rgb_value)

    print("==================================================================")
    print("gid_rgb_value:", gid_rgb_value)
    print("gid_rgb_mean_value:", [i / 255.0 for i in gid_rgb_value])

def dd_22():
    onnx_path = r'C:\Users\1\Desktop\STNet_opset12.onnx'


def otsu_threshold(im):

    width, height = im.size
    img_size = width * height
    pixel_counts = np.zeros(256)
    for x in range(width):
        for y in range(height):
            pixel = im.getpixel((x, y))
            pixel_counts[pixel] = pixel_counts[pixel] + 1
    # 得到图片的以0-255索引的像素值个数列表
    s_max = (0, -10)
    for threshold in range(256):
        # 遍历所有阈值
        # 更新
        n_0 = sum(pixel_counts[:threshold])  # 得到阈值以下像素个数
        n_1 = sum(pixel_counts[threshold:])  # 得到阈值以上像素个数

        w_0 = n_0 / img_size
        w_1 = n_1 / img_size
        # 得到阈值下所有像素的平均灰度
        u_0 = sum([i * pixel_counts[i] for i in range(0, threshold)]) / n_0 if n_0 > 0 else 0

        # 得到阈值上所有像素的平均灰度
        u_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / n_1 if n_1 > 0 else 0

        # 总平均灰度
        u = w_0 * u_0 + w_1 * u_1

        # 类间方差
        g = w_0 * (u_0 - u) * (u_0 - u) + w_1 * (u_1 - u) * (u_1 - u)

        # 类间方差等价公式
        # g = w_0 * w_1 * (u_0 * u_1) * (u_0 * u_1)

        # 取最大的
        if g > s_max[1]:
            s_max = (threshold, g)
    return s_max[0]

def batch_ostu():
    img_dir = r'C:\Users\1\Desktop\wood_Counting_patent\dataset'
    result_dir = r'C:\Users\1\Desktop\wood_Counting_patent\result'
    for img_path in os.listdir(img_dir):
        img = Image.open(os.path.join(img_dir, img_path)).convert('L')
        adaptive_threshold = otsu_threshold(img)
        img_array = np.array(img)
        img_array[img_array >= adaptive_threshold] = 255
        img_array[img_array < adaptive_threshold] = 0
        result = Image.fromarray(img_array)
        result.save(os.path.join(result_dir, img_path))


import numpy as np
import cv2


def bgr2nv21():
    bgr = cv2.imread(r'F:\BatchSmokeClassification\x64\Release\neg_data\619.jpg')
    bgr = cv2.resize(bgr, (224, 224))

    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    height = bgr.shape[0]
    width = bgr.shape[1]

    u = i420[height: height + height // 4, :]
    u = u.reshape((1, height // 4 * width))
    v = i420[height + height // 4: height + height // 2, :]
    v = v.reshape((1, height // 4 * width))
    uv = np.zeros((1, height // 4 * width * 2))
    uv[:, 0::2] = v
    uv[:, 1::2] = u
    uv = uv.reshape((height // 2, width))
    nv21 = np.zeros((height + height // 2, width))
    nv21[0:height, :] = i420[0:height, :]
    nv21[height::, :] = uv
    #return nv21
    nv21.astype("uint8").tofile(r"C:\Users\1\Desktop\tmp\tmp_20230508\619_uint8.jpg.yuv")

def dd_23():
    array = np.zeros([576, 1024])
    img = Image.fromarray(array, "RGB")
    img.save(r'C:\Users\1\Desktop\tmp\tmp_20230508\0.jpg')

    array = np.ones([576, 1024, 3])* 255

    cv2.imwrite(r'C:\Users\1\Desktop\tmp\tmp_20230508\255.jpg',array, )

def dd_24():
    yuv_path = r'C:\Users\1\Desktop\tmp\tmp_20230508\yuv_sp420_224_224.sp420'
    f = open(yuv_path, 'rb')
    b_img = f.readlines()
    f.close()
    print(len(b_img))
    print(b_img[0])
    # img = Image.open(io.BytesIO(b_img[0]))
    # print(img)
    # cv::COLOR_YUV420sp2RGB = COLOR_YUV2RGB_NV21,
    # cv::COLOR_YUV420sp2BGR = COLOR_YUV2BGR_NV21,

    bgr = cv2.cvtColor(b_img[0], cv2.COLOR_YUV2BGR_NV21)

    cv2.imwrite(r'C:\Users\1\Desktop\tmp\tmp_20230508\yuv_sp420_224_224_sp420_bgr.jpg', bgr)

def process(input_path):
    try:
        input_image = Image.open(input_path)
        input_image = input_image.resize((224, 224))
        # hwc
        img = np.array(input_image)
        height = img.shape[0]
        width = img.shape[1]
        # h_off = int((height-224)/2)
        # w_off = int((width-224)/2)
        # crop_img = img[h_off:height-h_off, w_off:width-w_off, :]
        # # rgb to bgr
        # img = crop_img[:, :, ::-1]
        shape = img.shape
        #img = img.astype("float16")
        img = img.astype("float32")

        img = img[:, :, ::-1]

        img[:, :, 0] -= 124
        img[:, :, 1] -= 133
        img[:, :, 2] -= 134
        # img /= 255.0

        img = img.reshape([1] + list(shape))
        result = img.transpose([0, 3, 1, 2])
        print(result.shape)
        output_name = os.path.join(r'C:\Users\1\Desktop\tmp\tmp_20230508\resnet_neg_bin', input_path.split('\\')[-1].split('.')[0] + ".bin")
        #output_name = r'C:\Users\1\Desktop\tmp\tmp_20230508\127_float32.bin'
        result.tofile(output_name)
    except Exception as except_err:
        print(except_err)
        return 1
    else:
        return 0

def dd_25():
    img = np.zeros([1,3,224,224], dtype= 'float32')
    img.tofile(r'C:\Users\1\Desktop\tmp\tmp_20230508\all_0.bin')

    img = np.ones([1,3,224,224], dtype= 'float32')
    img.tofile(r'C:\Users\1\Desktop\tmp\tmp_20230508\all_1.bin')

def compute_roundness(label_image):
    import math
    contours, hierarchy = cv2.findContours(np.array(label_image, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(contours)
    print("=========================")
    print(hierarchy)

    a = cv2.contourArea(contours[0]) * 4 * math.pi
    b = math.pow(cv2.arcLength(contours[0], True), 2)
    if b == 0:
        return 0
    return a / b

def dd_26():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 1], [0, 0]], dtype= bool)

    c = np.mean(a,where=b)
    print(c)

def dd_27():
    lb_file = r'C:\Users\1\Desktop\jlbx-220501_1661584684374.txt'
    with open(lb_file) as f:
        lb = [x.split() for x in f.read().strip().splitlines() if len(x) and x[0] == '0']
    print(lb)

def dd_28():
    img_dir = r'E:\pycode\datasets\DZYH_test_dataset\yzw20220500_test_smoke\images'
    for file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, file)
        new_path = os.path.join(img_dir, file.split('.mp4')[-1][1:])
        shutil.copy(img_path, new_path)

def dd_29():
    img_dir = r'F:\Hi3559A\nnie\windows\RuyiStudio-2.0.38\workspace\3559Box\image\fn_file'
    for file in [i for i in os.listdir(img_dir) if i.endswith('jpg')]:
        img_path = os.path.join(img_dir, file)
        img = cv2.imread(img_path)
        n_img = cv2.resize(img, (1024, 576))
        cv2.imwrite(os.path.join(r'F:\Hi3559A\nnie\windows\RuyiStudio-2.0.38\workspace\3559Box\image\fn_file', file), n_img)


def dd_30():
    import cv2
    import numpy as np

    """
    使用单高斯模型检测静止相机下的运动目标
    """
    np.set_printoptions(precision=2, suppress=True)
    # 设置常数
    T = 30000  # 前后景区分常数
    lr = 1  # 学习率

    # 读取视频
    cap = cv2.VideoCapture(r'E:\cppcode\MotionDetectionImageData\video\posTest_tmp\0608-1017.mp4')
    isFirst = True

    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[540:, :960, :]
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # 使用第一帧来初始化参数
            if isFirst:
                mean = np.zeros(gray.shape)
                var = np.zeros(gray.shape)
                isFirst = False
            else:
                # 标识目标
                frame[(gray - mean) ** 2 > T * var, 2] = 255
                cv2.imshow('tracking', frame)
                # 更新参数
                mean = lr * mean + (1 - lr) * gray
                var = lr * var + (1 - lr) * (gray - mean) ** 2
                if cv2.waitKey(50) and 0xFF == ord('q'):
                    break
        else:
            break

def dd_31():
    img_path = r'Z:\qdm\Potsdam\RGB图片和标签，分割与未分割\2_Ortho_RGB-未分割的大图\top_potsdam_2_10_RGB.tif'
    #image_numpy_data = cv2.imdecode(np.fromfile(image_file_name, dtype=np.uint8), -1)
    img = cv2.imdecode(np.fromfile(img_path, dtype = np.uint8), -1)
    print(img.shape)
    print(np.unique(img[:100,:100,:]))

def dd_32():
    import json
    json_path = r'../LCC5C_b512_woOverlap_test.json'
    with open(json_path, 'r') as load_f:
        dict_dataset = json.load(load_f)
    print(len(dict_dataset))
    print(dict_dataset[str(0)])
    print(dict_dataset[str(0)]['imagePath'])
    print(dict_dataset[str(0)]['x'])
    print(dict_dataset[str(0)]['y'])

#{'imagePath': 'D:\\zz\\GID\\LCC5C\\image_RGB_test\\GF2_PMS2__L1A0001574001-MSS2.tif', 'labelPath': 'D:\\zz\\GID\\LCC5C\\label_5classes_test\\GF2_PMS2__L1A0001574001-MSS2_label.tif', 'x': 0, 'y': 0, 'block_x': 512, 'block_y': 512, 'width': 7200, 'height': 6800}
def dd_33():
    import json
    a = "image:{}".format("hh")
    print(a)
    key = ['imagePath', 'labelPath', 'x', 'y', 'block_x', 'block_y', 'width', 'height']
    all_dict = {}
    for i in range(3):
        img_path = "/home/qudaming/image/{}".format(i)
        lab_path = "/home/qudaming/label/{}".format(i)
        x = i*10
        y = i*100
        w = i+1
        h = i+2
        values = [img_path, lab_path,x, y, w, h]
        my_dict = dict(zip(key, values))
        all_dict.setdefault("{}".format(i), my_dict)
    print(all_dict)
    j_str = json.dumps(all_dict)
    with open ("../temp_json.json", 'a') as f:
        f.write(j_str)



if __name__ == '__main__':
    dd_33()