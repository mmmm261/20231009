import glob
import shutil
import sqlite3

import cv2
from xml.dom import minidom
import numpy as np
import os

class VOC_Sample_Generator:

    def __init__(self):
        self.dom = minidom.Document()
        self.root_node = self.dom.createElement('annotation')
        #demo1依赖
        #self.object_node = self.dom.createElement('object')
        #self.root_node.appendChild(self.object_node)

    #demo2依赖
    def add_object(self):
        self.object_node = self.dom.createElement('object')
        self.root_node.appendChild(self.object_node)

    def add_folder(self, folder):
        text = self.dom.createTextNode(folder)
        folder_node = self.dom.createElement('folder')
        folder_node.appendChild(text)
        self.root_node.appendChild(folder_node)

    def add_filename(self, filename):
        text = self.dom.createTextNode(filename)
        filename_node = self.dom.createElement('filename')
        filename_node.appendChild(text)
        self.root_node.appendChild(filename_node)

    def add_path(self, path):
        text = self.dom.createTextNode(path)
        path_node = self.dom.createElement('path')
        path_node.appendChild(text)
        self.root_node.appendChild(path_node)

    def add_database(self, database):
        source_node = self.dom.createElement('source')
        text = self.dom.createTextNode(database)
        database_node = self.dom.createElement('database')
        database_node.appendChild(text)
        source_node.appendChild(database_node)
        self.root_node.appendChild(source_node)

    def add_size(self, width, height):
        text = self.dom.createTextNode(str(width))
        width_node = self.dom.createElement('width')
        width_node.appendChild(text)

        text = self.dom.createTextNode(str(height))
        height_node = self.dom.createElement('height')
        height_node.appendChild(text)

        # text = self.dom.createTextNode(str(depth))
        # depth_node = self.dom.createElement('depth')
        # depth_node.appendChild(text)

        size_node = self.dom.createElement('size')

        size_node.appendChild(width_node)
        size_node.appendChild(height_node)
        #size_node.appendChild(depth_node)
        self.root_node.appendChild(size_node)

    def add_segmented(self, segmented):
        text = self.dom.createTextNode(segmented)
        segmented_node = self.dom.createElement('segmented')
        segmented_node.appendChild(text)
        self.root_node.appendChild(segmented_node)

    #def add_nptd(self,name,pose,truncated,difficult):
    def add_nptd(self,name,trackid,occluded,generated,x1,y1,x2,y2):
        name_text = self.dom.createTextNode(name)
        #pose_text = self.dom.createTextNode(pose)
        track_text = self.dom.createTextNode(trackid)
        occluded_text = self.dom.createTextNode(occluded)
        generated_text = self.dom.createTextNode(generated)
        name_node = self.dom.createElement('name')
        name_node.appendChild(name_text)
        # pose_node = self.dom.createElement('pose')
        # pose_node.appendChild(pose_text)
        track_node = self.dom.createElement('trackid')
        track_node.appendChild(track_text)
        occluded_node = self.dom.createElement('occluded')
        occluded_node.appendChild(occluded_text)
        generated_node = self.dom.createElement('generated')
        generated_node.appendChild(generated_text)

        self.object_node.appendChild(track_node)
        self.object_node.appendChild(name_node)

        self.add_bndbox(x1,y1,x2,y2)

        self.object_node.appendChild(occluded_node)
        self.object_node.appendChild(generated_node)

    def add_bndbox(self, xmin, ymin, xmax, ymax):
        text = self.dom.createTextNode(str(xmin))
        xmin_node = self.dom.createElement('xmin')
        xmin_node.appendChild(text)

        text = self.dom.createTextNode(str(ymin))
        ymin_node = self.dom.createElement('ymin')
        ymin_node.appendChild(text)

        text = self.dom.createTextNode(str(xmax))
        xmax_node = self.dom.createElement('xmax')
        xmax_node.appendChild(text)

        text = self.dom.createTextNode(str(ymax))
        ymax_node = self.dom.createElement('ymax')
        ymax_node.appendChild(text)

        bndbox_node = self.dom.createElement('bndbox')

        bndbox_node.appendChild(xmax_node)
        bndbox_node.appendChild(xmin_node)
        bndbox_node.appendChild(ymax_node)
        bndbox_node.appendChild(ymin_node)

        self.object_node.appendChild(bndbox_node)

    def build(self, path):
        self.dom.appendChild(self.root_node)
        with open(path, 'w') as f:
            self.dom.writexml(f, indent='', addindent='\t', newl='\n', encoding='UTF-8')
        f.close()

def get_HWC(img_path):
    try:
        return cv2.imdecode(np.fromfile(img_path, np.uint8), flags = cv2.IMREAD_COLOR).shape
    except:
        print("get_HWC_error, and return (480, 480, 3)")
        return (1920, 1080, 3)

def get_xyxy(txt_path, W, H):
    with open(txt_path, 'r') as f:
        files = f.readlines()
    point_list = []
    for lines in files:
        line = [float(i) for i in lines.split()]
        try:
            cla, x, y, w, h, conf = line
        except:
            cla, x, y, w, h = line
        if cla == 0:
            xx = x * W
            yy = y * H
            hh = h * H
            ww = w * W
            min_left = int(xx - ww / 2)
            min_top = int(yy - hh / 2)
            max_right = int(xx + ww / 2)
            max_down = int(yy + hh / 2)
            point_list.append([cla, min_left, min_top, max_right, max_down])
    #print(len(point_list))
    return point_list

def video_file2img_file():
    video_dir = r'F:\dataset\smoke_detection_video\cc20220819\Data'
    img_dir = r'F:\dataset\smoke_detection_video\cc20220819\ImgData'
    txt_dir= r'F:\dataset\smoke_detection_video\cc20220819\txt'
    video_name_list = os.listdir(video_dir)
    for i in video_name_list:
        video_path = os.path.join(video_dir, i)
        img_path = os.path.join(img_dir, i.split('.')[0])

        if os.path.exists(os.path.join(txt_dir, i.replace('mp4', 'txt'))):
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            cap = cv2.VideoCapture(video_path)
            j = 0
            while cap.isOpened():
                ret, img = cap.read()
                if ret:
                    img_name = os.path.join(img_path, str(j).zfill(6) + ".jpg")
                    cv2.imwrite(img_name, img)
                    j += 1
                else:
                    break

def generate_smoke_xml(img_path, txt_path, save_dir, cls_name):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_list = os.listdir(img_path)  # os.path.join(result_path + '\\*.jpg')
    H, W, _ = get_HWC(os.path.join(img_path, file_list[0]))
    print("txt_path:", txt_path)
    points = get_xyxy(txt_path, W, H) if os.path.exists(txt_path) else []
    #print("points:", points)
    for file in file_list:
        print(file)
        if file.endswith('jpg'):
            xml_path = os.path.join(save_dir, file.replace('jpg', 'xml'))
            print("xml_path:", xml_path)
            #continue
            if points:
                voc = VOC_Sample_Generator()
                voc.add_folder('cc20220819')
                voc.add_filename(file.split('.')[0])
                voc.add_database('ILSVRC_2015')
                voc.add_size(W, H)
                for point in points:
                    voc.add_object()
                    voc.add_nptd(name=cls_name, trackid='0', occluded='1', generated='0', x1=point[1], y1=point[2], x2=point[3], y2=point[4])
                voc.build(xml_path)

def batch_generate_smoke_xml():
    img_dir = r'F:\dataset\smoke_detection_video\cc20220819\ImgData'
    txt_dir = r'F:\dataset\smoke_detection_video\cc20220819\SingleImageLabel'
    save_dir = r'F:\dataset\smoke_detection_video\cc20220819\ImgAnnotations'
    class_name = 'smoke'

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(txt_dir, img_name + ".txt")
        save_path = os.path.join(save_dir, img_name)
        generate_smoke_xml(img_path, txt_path, save_path, class_name)

def generate_imgList_npy():
    data_dir = r'F:\dataset\smoke_detection_video\cc20220819'
    img_dir = os.path.join(data_dir, "ImgData")
    npy_path = os.path.join(data_dir, os.path.basename(data_dir) + '.npy')
    img_list = []
    empty_list = dd_2()
    for img_path in os.listdir(img_dir):
        print(img_path)
        if img_path not in empty_list:
            img_file_list = glob.glob(os.path.join(img_dir, img_path + "\\*.jpg"))
            #ubuntu
            file_list = ['/'.join(i.split('\\')[-4:]) for i in img_file_list]
            #windows
            #file_list = [os.path.join(*i.split('\\')[-4:]) for i in img_file_list]
            print(file_list)
            img_list.append(file_list)
    img_name_array = np.array(img_list, dtype = object)
    np.save(npy_path, img_name_array)

def assert_imgList_npy():
    npy_path = r'F:\dataset\smoke_detection_video\cc20220819\cc20220819.npy'
    dataset = np.load(npy_path, allow_pickle=True).tolist()
    print(len(dataset))
    print(dataset[0])
    print(dataset[:2])
    print(dataset[-1])

def dd():
    anno_res = []
    name_list = ['smoke']
    name_num = [0]
    xmls = r'F:\dataset\smoke_detection_video\cc20220819\ImgAnnotations\fs-20200610_1659519758513\000000.xml'
    file = minidom.parse(xmls)
    root = file.documentElement
    objs = root.getElementsByTagName("object")
    width = int(root.getElementsByTagName('width')[0].firstChild.data)
    height = int(root.getElementsByTagName('height')[0].firstChild.data)
    tempnode = []

    for obj in objs:
        nameNode = obj.getElementsByTagName("name")[0].firstChild.data
        xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
        xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
        ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
        ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
        x1 = np.max((0, xmin))
        y1 = np.max((0, ymin))
        x2 = np.min((width, xmax))
        y2 = np.min((height, ymax))
        if x2 >= x1 and y2 >= y1:
            print(0)
            # tempnode.append((name_num[nameNode],x1,y1,x2,y2,))
            tempnode.append((x1, y1, x2, y2, 0))

    num_objs = len(tempnode)
    # print("num_objs:", num_objs)
    res = np.zeros((num_objs, 5))
    r = min(512 / height, 512 / width)
    for ix, obj in enumerate(tempnode):
        res[ix, 0:5] = obj[0:5]
    res[:, :-1] *= r
    print(res)
    anno_res.append(res)
    print(anno_res)

def dd_1():
    import cv2
    img_path = r'F:\dataset\smoke_detection_video\cc20220819\ImgData\hnh20210917_1660468821522/000131.jpg'
    txt_path = r'F:\dataset\smoke_detection_video\cc20220819\SingleImageLabel\hnh20210917_1660468821522.txt'
    img = cv2.imread(img_path)
    label = [0.120573, 0.169444, 0.057813, 0.088889]
    width = 1920
    height = 1080
    label[0] *= width
    label[1] *= height
    label[2] *= width
    label[3] *= height
    print(label)
    origal_points = [98, 239, 161, 410]
    #x1, y1, x2, y2 = origal_points
    #x1, y1, x2, y2 = label

    x,y,w,h = label
    x1 = int(x - w*0.5)
    y1 = int(y - h*0.5)
    x2 = int(x + w*0.5)
    y2 = int(y + h*0.5)
    print(x1, y1, x2, y2)

    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 0)

    cv2.imwrite(r"E:\pycode\YOLOV-master\result\label_xy.jpg", img)

    p,t,z = get_HWC(img_path)
    print(p,t,z)
    p = get_xyxy(txt_path, width, height)
    print(p)
    p1 = [[0.0, 175, 134, 287, 230]]

def dd_2():
    empty_list = []
    dir = r'F:\dataset\smoke_detection_video\cc20220819\ImgAnnotations'
    path_list = os.listdir(dir)
    for path in path_list:
        length = len(os.listdir(os.path.join(dir, path)))
        if length == 0:
            print(path)
            empty_list.append(path)
    return empty_list

def check_label():
    txt_dir = r'F:\dataset\smoke_detection_video\cc20220819\SingleImageLabel'
    i = 0
    j = 0
    for file in os.listdir(txt_dir):
        txt_path = os.path.join(txt_dir, file)
        with open(txt_path, 'r') as f:
            files = f.readlines()
        point_list = []
        for lines in files:
            line = [float(i) for i in lines.split()]
            cla, x, y, w, h = line
            if cla == 0:
                i += 1
            elif cla == 1:
                j += 1
                print("background")
    print(i)
    print(j)

def dd_3():
    data_path = r'F:\dataset\smoke_detection_video\cc20220819\ImgData'
    data_list = []
    for imgName in os.listdir(data_path):
        img_path = os.path.join(data_path, imgName)
        img_list = [os.path.join(img_path, i) for i in os.listdir(img_path) if (int(i[:6]) + 1) % 8 == 0]
        data_list += img_list
    with open(r'F:\dataset\smoke_detection_video\cc20220819\testlist.txt', 'a') as f:
        for file in data_list[:1000]:
            f.writelines(file + '\n')

def dd_4():
    for i in reversed(range(4)):

        j = 40 - i*5
        print(j)

def dd_5():
    img_dir = r'E:\pycode\datasets\DZYH_test_dataset\yzw20220500_test_smoke\images'
    for imgName in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, imgName))
        new_img = cv2.resize(img, (512, 288))
        save_path = r"E:\pycode\datasets\DZYH_test_dataset\tmp_288_512\{}".format(imgName)
        cv2.imwrite(save_path, new_img)
if __name__ == '__main__':
    #F:\BaiduNetdiskDownload\ILSVRC2015\Annotations\VID\train\ILSVRC2015_VID_train_0000\ILSVRC2015_train_00010000
    # generate_imgList_npy()
    # assert_imgList_npy()
    # dd_1()
    #batch_generate_smoke_xml()
    # dd_2()
    #check_label()
    #dd_4()
    dd_5()
