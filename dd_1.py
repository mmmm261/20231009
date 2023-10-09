import glob
import os
import torch
import io
import yaml
import cv2
import numpy as np
import math
import shutil

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10
def test():
    m = torch.jit.script(MyModule())

    # Save to file
    torch.jit.save(m, 'scriptmodule.pt')
    # This line is equivalent to the previous
    m.save("scriptmodule.pt")

    # Save to io.BytesIO buffer
    buffer = io.BytesIO()
    torch.jit.save(m, buffer)

    # Save with extra files
    extra_files = {'foo.json': b'bar'}
    torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)

def test_2():
    weight_path = r'E:\cppcode\weight_backup_standard\epoch50_pretrained_20220811_best.pt'
    model = torch.load(weight_path, map_location= 'cpu')
    dummy_input = torch.randn(1,3,1024,1024).to('cpu')
    torch_trace_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(torch_trace_model, 'E:\cppcode\weight_backup_standard\wuhu.pt')

def test_3():
    hyp_path= ''
    with open(hyp_path, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict


    weights = r'E:\cppcode\weight_backup_standard\epoch50_pretrained_20220811_best.pt'
    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys

        model.load_state_dict(model.state_dict(), strict=False)  # load

def dd():
    img_path = r'C:\Users\1\Desktop\3559_cloud_reference\weights\quantization_data'
    new_img_path = r'C:\Users\1\Desktop\3559_cloud_reference\weights\quantization_data_mini'
    for file in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path, file))
        new_img = cv2.resize(img, (1024, 576))
        cv2.imwrite(os.path.join(new_img_path, file), new_img)

def dd_1():
    input_list = [i/10 for i in range(10)]
    print(input_list)
    for input in input_list:
        output = torch.sigmoid(torch.tensor(input))
        print(input, ":", output.numpy())

def letter_box(img, letter_box_size):
    img_height, img_width, _ = img.shape
    mask_size = max(img_height, img_width)
    print(mask_size)
    mask = np.zeros((mask_size, mask_size, 3))
    if img_width > img_height:
        idx = int((img_width - img_height) / 2)
        mask[idx: idx + img_height, :, :] = img
    else:
        idx = int((img_height - img_width) / 2)
        mask[:, idx: idx + img_width, :] = img
    if mask_size != letter_box_size:
        mask = cv2.resize(mask, (letter_box_size, letter_box_size))
    return mask

def dd_2():
    video_path = r'F:\Hi3403\smoke_detection_code\DependentFile\fs-20200610_1653444770674.mp4'
    save_path = r'F:\Hi3403\smoke_detection_code\DependentFile\test2'
    videoCapture = cv2.VideoCapture(video_path)
    i = 0
    while True:
        new_path =  os.path.join(save_path, "stepone_source_image{}.jpg".format(int(i/3)))
        success, frame = videoCapture.read()
        if i%3 == 0:
            cv2.imwrite(new_path, frame)
        i += 1
        if i > 40*3:
            break

def dd_3():
    path = r'E:\Qtcode\3403_PC_AddPC_Infer\3403_PC_AddPC_Infer\x64\Release\DependentFile\tmp\test22.jpg'
    img = cv2.resize(cv2.imread(path),(10,5))
    cv2.imwrite(r'E:\Qtcode\3403_PC_AddPC_Infer\3403_PC_AddPC_Infer\x64\Release\DependentFile\tmp\test22_s.jpg', img)

def dd_4():
    tif_dir_list = []
    for dir, path, file in os.walk(r'F:\ir_tmp'):
        for tif_path in file:
            if tif_path.endswith("tiff"):
                tif_dir_list.append(dir)
                break
    print(len(tif_dir_list))
    for i, tif_dir in enumerate(tif_dir_list):
        if i > 120:
            for tif_file in os.listdir(tif_dir):
                if tif_file.endswith("tiff"):
                    file_path = os.path.join(tif_dir, tif_file)
                    print(file_path)
                    try:
                        tif = cv2.imdecode(np.fromfile(file_path, dtype=np.uint16), -1)
                        new_tif = trans16To8(tif)
                        save_path = file_path.replace('.tiff', '.png')
                        print(save_path)
                        print(np.max(tif), np.min(tif))
                        cv2.imencode('.png', new_tif)[1].tofile(save_path)
                    except:
                        print("-----------------")
                        print("To do:", file_path)
                        print("-----------------")

def trans16To8(img):
    min_value = np.min(img)
    max_value = np.max(img)
    print("before:", max_value, min_value)
    new_img = 255 * ((img - min_value) / (max_value - min_value))
    print("after:", np.max(new_img), np.min(new_img))
    #print(max_value - min_value, (max_value - min_value) / (max_value - min_value),  255 * (max_value - min_value))
    #print(max_value - min_value, 255 * ((max_value - min_value) / (max_value - min_value)),  255 * (max_value - min_value)/(max_value - min_value))
    return new_img

def dd_5():
    for i, file in enumerate(os.listdir(r'E:\pycode\datasets\tmp\image')):
        tif_path = os.path.join(r'E:\pycode\datasets\tmp\image', file)
        tif = cv2.imdecode(np.fromfile(tif_path, dtype= np.uint16), -1)
        print(tif.shape)
        print(tif_path)
        print(np.max(tif), np.min(tif))
        print("--------------------------")
        new_tif = trans16To8(tif)
        cv2.imwrite(tif_path.replace('image', 'result').replace('tiff', 'png'), new_tif)


def dd_6():
    import cv2
    import numpy as np

    # 读入原始图像
    img = cv2.imread(r'C:\Users\1\Desktop\1.jpg')

    print(img.shape)
    # 提取水印区域
    watermark = img[550:, 870:, :]

    print(watermark.shape)

    # 使用高斯滤波进行平滑处理
    watermark = cv2.GaussianBlur(watermark, (3, 3), 0)

    # 对原始图像进行匹配
    res = cv2.matchTemplate(img, watermark, cv2.TM_CCOEFF_NORMED)

    # 设定匹配阈值
    threshold = 0.2

    # 获取匹配结果
    loc = np.where(res >= threshold)
    print(loc)
    # 在匹配位置覆盖原始图像
    for pt in zip(*loc[::-1]):
        print(pt)
        #img[pt[1]:pt[1] + watermark.shape[0], pt[0]:pt[0] + watermark.shape[1]] = (255, 255, 255)
        img[550:, 870:, :] = img[pt[1]:pt[1] + watermark.shape[0], pt[0]:pt[0] + watermark.shape[1]]
        break
        #img[pt[1]:pt[1] + watermark.shape[0], pt[0]:pt[0] + watermark.shape[1]] = (255, 255, 255)

    # 显示结果
    cv2.imshow('Result', img)
    cv2.waitKey(0)

def dd_7():
    for path in os.listdir(r'F:\ir_tmp\测试6-陕西测试-2019.1.9-2019.1.10（RAW和TIFF）\测试5-安康市香溪隧道\tmp'):
        file_path = os.path.join(r'F:\ir_tmp\测试6-陕西测试-2019.1.9-2019.1.10（RAW和TIFF）\测试5-安康市香溪隧道\tmp', path)
        png_path =[i for i in os.listdir(file_path) if i.endswith('png')]
        tif_path =[i for i in os.listdir(file_path) if i.endswith('tiff')]
        for i in tif_path:
            img_path = os.path.join(file_path, i)
            print(img_path)
            img = cv2.imdecode(np.fromfile(img_path, dtype= np.uint16), -1)
            png_img = trans16To8(img)
            print(np.max(png_img))
            print("png_img_path:", img_path.replace('tiff', 'png'))
            #cv2.imencode('.png', png_img)[1].tofile(img_path.replace('tiff', 'png'))
            print("======================")

def dd_8():
    tif_path = r'F:\ir_tmp\测试6-陕西测试-2019.1.9-2019.1.10（RAW和TIFF）\测试5-安康市香溪隧道\tmp\测试1-690后截止\Pic_2019_01_10_111138_blockId#5059.tiff'
    tif_img = cv2.imdecode(np.fromfile(tif_path, dtype= np.uint16), -1)
    png_img = trans16To8(tif_img)
    print("============")

def dd_9():
    mask = np.ones([1080,1920]) * 255
    mask[0:150,0:860] = 0
    cv2.imwrite(r'C:\Users\1\Desktop\tmp\shield.bmp', mask)

def dd_10():
    import py_compile
    py_compile.compile(r'/download_video_from_web.py')

def dd_11():
    file_path = r'F:\download_web_video_tmp\2_month.txt'
    with open(file_path, 'r') as f:
        file_list = f.readlines()
    print(len(file_list))

def one_cycle(x, y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

def dd_12():
    dir_list = os.listdir(r'F:\tmp_single_frame_data_after_cls')
    for path in dir_list:
        new_image_path = r"F:\tmp_single_frame_data_after_cls\{}\images".format(path)
        ori_label_path = r"F:\tmp_single_frame_data\{}\labels".format(path)
        new_label_path = r"F:\tmp_single_frame_data_after_cls\{}\labels".format(path)
        if not os.path.exists(new_label_path):
            os.mkdir(new_label_path)
        for img_file in os.listdir(new_image_path):
            lab_file = "{}.txt".format(img_file.split('.')[0])
            old_lab_path = os.path.join(ori_label_path, lab_file)
            new_lab_path = os.path.join(new_label_path, lab_file)

            if os.path.exists(old_lab_path):
                if not os.path.exists(new_lab_path):
                    shutil.copy(old_lab_path, new_lab_path)
                    #print(old_lab_path)
                    #print(new_lab_path)

def dd_13():
    img_path = r'F:\tmp_single_frame_data_after_cls\cc20220919\images'
    txt_path = r'F:\tmp_single_frame_data_after_cls\cc20220919\20220811_best_pretrained_20220830_best_result_txt'
    xml_path = r'F:\tmp_single_frame_data_after_cls\cc20220919\xmls'
    file_list = [i.split('.')[0] for i in os.listdir(img_path)]
    print(len(file_list))
    for file in file_list:
        txt_file = os.path.join(txt_path, "{}.txt".format(file))
        xml_file = os.path.join(xml_path, "{}.xml".format(file))
        if os.path.exists(txt_file):
            new_txt_path = r"F:\tmp_single_frame_data_after_cls\cc20220919\new_txt\{}.txt".format(file)
            shutil.copy(txt_file, new_txt_path)

        if os.path.exists(xml_file):
            new_xml_path = r"F:\tmp_single_frame_data_after_cls\cc20220919\new_xmls\{}.xml".format(file)
            shutil.copy(xml_file, new_xml_path)

def dd_14():
    import ffmpy  # 导入
    fileOldPath = "F:\wchat_voice_tmp\msg_411338052121142654764e1102.amr"  # arm文件地址
    fileNewPath = "F:\wchat_voice_tmp\msg_411338052121142654764e1102.mp3"  # 转换后MP3文件地址
    ff = ffmpy.FFmpeg(
        inputs={fileOldPath: None},
        outputs={fileNewPath: None}
    )
    ff.run()

def dd_15():
    dir = r'F:\wchat_voice_tmp\voice_file\huawei_voice'
    amr_dir = r'F:\wchat_voice_tmp\voice_file\huawei_voice_collection'
    amr_list = []
    for root, path, file in os.walk(dir):
        if len(file) > 0:
            amr_list.append(root)
    print(len(amr_list))
    for amr_path in amr_list:
        amr_file = [i for i in os.listdir(amr_path) if i.endswith('.amr')]
        for amr_file_ in amr_file:
            old_amr_path = os.path.join(amr_path, amr_file_)
            new_amr_path = os.path.join(amr_dir, amr_file_)
            shutil.copy(old_amr_path, new_amr_path)

def dd_16():
    img_path = r'C:\Users\1\Desktop\tmp\MACU-Net_result\GF2_PMS2__L1A0001838560-MSS2_5_9.png'
    img = cv2.imread(img_path)

    kernel = np.ones((3, 3), dtype=np.uint8)

    dilate = cv2.dilate(img, kernel, 1)  # 1:迭代次数，也就是执行几次膨胀操作
    result = cv2.erode(dilate, kernel, iterations=1)

    #erode =  cv2.erode(img, kernel, iterations=1)
    #result =cv2.dilate(erode, kernel, 1)

    cv2.imwrite(r'C:\Users\1\Desktop\tmp\MACU-Net_result\GF2_PMS2__L1A0001838560-MSS2_5_9.jpg', result)


def dd_17():
    import imageio.v2 as imageio

    semantic_image_path = r'C:\Users\1\Desktop\tmp\MACU-Net_result\MACUNet_1129_1_idx3and5_iou0.76\GF2_PMS1__L1A0000647767-MSS1_12_11.png'
    semantic_image = torch.from_numpy(imageio.imread(semantic_image_path, as_gray=True) / 51.0).type(torch.FloatTensor)
    semantic_image = torch.squeeze(semantic_image, 0)
    print(np.unique(semantic_image))

    label_image_path = r'C:\Users\1\Desktop\tmp\MACU-Net_result\lab_idx_3and5\GF2_PMS1__L1A0000647767-MSS1_label_12_11.png'
    label_image = torch.from_numpy(imageio.imread(label_image_path, as_gray=True) / 51.0).type(torch.FloatTensor)
    label_image = torch.squeeze(label_image, 0)
    print(np.unique(label_image))

    confusionMatrix = np.zeros((6) * 2)
    print(confusionMatrix)

    mask = (label_image >= 0) & (label_image < 6)
    label = 6 * label_image[mask] + semantic_image[mask]
    print(np.unique(mask))
    print(np.unique(label))
    print(mask.shape)
    print(label_image[mask].shape)

    count = np.bincount(label, minlength=6 ** 2)

    print(count)
    confusionMatrix = count.reshape(6, 6)
    print(confusionMatrix)

    intersection = np.diag(confusionMatrix)
    print(intersection)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(confusionMatrix)
    print(union)
    IoU = intersection / union
    print(IoU)
    mIoU = np.nanmean(IoU)
    print(mIoU)

def dd_18():
    path = r'C:\Users\1\Desktop\000034.jpg'
    from PIL import Image
    img = Image.open(path)
    array = np.array(img)
    print(array.shape)
    cv2.imwrite(r'C:\Users\1\Desktop\000034_array.jpg', array)

def dd_19():
    dir = r'C:\Users\1\Desktop\animals'
    for path_ in os.listdir(dir):
        path = os.path.join(dir, path_)
        file_list = os.listdir(path)
        print(len(file_list))
        for file in file_list:
            if file.startswith('._'):
                print(os.path.join(path, file))
                os.remove(os.path.join(path, file))

def dd_20():
    im =cv2.imread(r'C:\Users\1\Desktop\assert_data\GF2_PMS1__L1A0000647767-MSS1_5_20.png')
    print(np.unique(im))
def dd_21():
    import numpy as np

    array = np.array([[3, 3, 9, 6, 4],
                      [2, 1, 9, 6, 3],
                      [2, 2, 8, 6, 3]])

    print(array.shape)

    value = np.sum(array, axis=1)
    print(value)
    value = np.sum(array, axis=0)
    print(value)
class demo(object):
    def __init__(self):
        self.A = "hello_world"
    def __str__(self):
        return self.A

def dd_22():
    img_list = glob.glob(os.path.join(r"F:\Hi3559A\nnie\windows\RuyiStudio-2.0.38\workspace\3559Box\image\data_640_640" + '/*.jpg'))

    for img_path in img_list:

        img = cv2.imread(img_path)
        n_img = cv2.resize(img, (640, 640))

        cv2.imwrite(img_path, n_img)

def dd_23():
    d_array = np.array(range(1*3*5*7*8)).reshape(1,3*5,7,8)
    d_tensor = torch.from_numpy(d_array)
    print(d_tensor)
    print("==================================")
    d_tensor =d_tensor.view(1,3,5,7,8).permute(0, 1, 3, 4, 2).contiguous()
    print(d_tensor)

def make_grid(nx=3, ny=7, i=0):
    anchors = torch.from_numpy(np.array([[10,13, 16,30, 33,23],[30,61, 62,45, 59,119], [116,90, 156,198, 373,326]]))
    stride = [32, 16, 8]
    na = 3
    d = anchors[i].device
    t = anchors[i].dtype
    shape = 1, na, ny, nx, 2  # grid shape
    y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)

    yv, xv = torch.meshgrid(y, x)
    grid = torch.stack((xv, yv), 2).expand(shape)# - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
    #anchor_grid = (anchors[i] * stride[i]).view((1, na, 1, 1, 2)).expand(shape)
    anchor_grid = (anchors[i]).view((1, na, 1, 1, 2)).expand(shape)
    print(grid[0,0,:,:,:])
    print(grid.shape)
    # print(anchor_grid[0, 0, :, :, :])
    # print(anchor_grid[0, 1, :, :, :])
    # print(anchor_grid[0, 2, :, :, :])
    print(anchor_grid.shape)

if __name__ == '__main__':
    make_grid()


