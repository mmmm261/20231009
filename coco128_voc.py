#yolov5的txt检测结果 转换为xml格式，用于斯总的数据标注，进行下一轮算法迭代。
import os
import sqlite3

import IPython.utils.path
import cv2
from xml.dom import minidom
import numpy as np

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

    def add_size(self, width, height, depth):
        text = self.dom.createTextNode(str(width))
        width_node = self.dom.createElement('width')
        width_node.appendChild(text)

        text = self.dom.createTextNode(str(height))
        height_node = self.dom.createElement('height')
        height_node.appendChild(text)

        text = self.dom.createTextNode(str(depth))
        depth_node = self.dom.createElement('depth')
        depth_node.appendChild(text)

        size_node = self.dom.createElement('size')

        size_node.appendChild(width_node)
        size_node.appendChild(height_node)
        size_node.appendChild(depth_node)
        self.root_node.appendChild(size_node)

    def add_segmented(self, segmented):
        text = self.dom.createTextNode(segmented)
        segmented_node = self.dom.createElement('segmented')
        segmented_node.appendChild(text)
        self.root_node.appendChild(segmented_node)

    def add_nptd(self,name,pose,truncated,difficult):
        name_text = self.dom.createTextNode(name)
        pose_text = self.dom.createTextNode(pose)
        truncated_text = self.dom.createTextNode(truncated)
        difficult_text = self.dom.createTextNode(difficult)
        name_node = self.dom.createElement('name')
        name_node.appendChild(name_text)
        pose_node = self.dom.createElement('pose')
        pose_node.appendChild(pose_text)
        truncated_node = self.dom.createElement('truncated')
        truncated_node.appendChild(truncated_text)
        difficult_node = self.dom.createElement('difficult')
        difficult_node.appendChild(difficult_text)
        self.object_node.appendChild(name_node)
        self.object_node.appendChild(pose_node)
        self.object_node.appendChild(truncated_node)
        self.object_node.appendChild(difficult_node)

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

        bndbox_node.appendChild(xmin_node)
        bndbox_node.appendChild(ymin_node)
        bndbox_node.appendChild(xmax_node)
        bndbox_node.appendChild(ymax_node)

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
        return (480,480,3)



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
# 测试

#one_obj_two_bnd
def demo1():
    result_path = r'F:\tmp_single_frame_data_after_cls\cc20230222\images'
    txt_path = r'F:\tmp_single_frame_data_after_cls\cc20230222\20220811_best_pretrained_20220830_best_result_txt'
    save_path = r'F:\tmp_single_frame_data_after_cls\cc20230222\20220811_best_pretrained_20220830_best_result_xml'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    file_list = os.listdir(result_path)  # os.path.join(result_path + '\\*.jpg')
    for file in file_list:
        print(file)
        if file.endswith('jpg'):
            points = []
            W, H, C = get_HWC(os.path.join(result_path, file))
            if os.path.exists(os.path.join(txt_path, file.replace('jpg', 'txt'))):
                # if os.path.exists(os.path.join(txt_path,"{}.txt".format(file))):
                points = get_xyxy(os.path.join(txt_path, file.replace('jpg', 'txt')), H, W)
            voc = VOC_Sample_Generator()

            voc.add_folder(r'images')
            voc.add_filename(file)
            voc.add_path(os.path.join(result_path, file))
            voc.add_database('Unknown')

            voc.add_size(H, W, C)
            voc.add_segmented('0')

            voc.add_nptd('smoke', 'Unspecified', '0', '0')

            if points:
                for point in points:
                    if point[0] == 0:
                        voc.add_bndbox(point[1], point[2], point[3], point[4])

            voc.build(os.path.join(save_path, file.replace('jpg', 'xml')))

#two obj_two_bnd usage!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def demo2():
    result_path = r'C:\Users\1\Desktop\yolov8_changed_dataset\changed\tmp_img0'
    txt_path = r'C:\Users\1\Desktop\yolov8_changed_dataset\origal\tmp_lab'
    save_path = r'C:\Users\1\Desktop\yolov8_changed_dataset\changed\tmp_xml0'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    file_list = os.listdir(result_path)  # os.path.join(result_path + '\\*.jpg')
    for file in file_list:
        print(file)
        if file.endswith('jpg'):
            points = []
            W, H, C = get_HWC(os.path.join(result_path, file))
            if os.path.exists(os.path.join(txt_path, file.replace('jpg', 'txt'))):
                # if os.path.exists(os.path.join(txt_path,"{}.txt".format(file))):
                points = get_xyxy(os.path.join(txt_path, file.replace('jpg', 'txt')), H, W)
            voc = VOC_Sample_Generator()

            voc.add_folder(r'images')
            voc.add_filename(file)
            voc.add_path(os.path.join(result_path, file))
            voc.add_database('Unknown')

            voc.add_size(H, W, C)
            voc.add_segmented('0')

            if points:
                for point in points:
                    if point[0] == 0:
                        voc.add_object()
                        voc.add_nptd('smoke', 'Unspecified', '0', '0')
                        voc.add_bndbox(point[1], point[2], point[3], point[4])

            voc.build(os.path.join(save_path, file.replace('jpg', 'xml')))

#result_path = r'D:\93_South_China_tiger'
#txt_path = r'C:\Users\1\Desktop\tmp_tiger_result\labels'
#save_path = r'C:\Users\1\Desktop\tmp_tiger_result\xmls_2'
def demo_for_animal(result_path, txt_path, save_path, cls_name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    file_list = os.listdir(result_path)  # os.path.join(result_path + '\\*.jpg')
    for file in file_list:
        print(file)
        if file.endswith('jpg'):
            points = []
            W, H, C = get_HWC(os.path.join(result_path, file))
            if os.path.exists(os.path.join(txt_path, file.replace('jpg', 'txt'))):
                # if os.path.exists(os.path.join(txt_path,"{}.txt".format(file))):
                points = get_xyxy(os.path.join(txt_path, file.replace('jpg', 'txt')), H, W)
            voc = VOC_Sample_Generator()

            voc.add_folder(r'images')
            voc.add_filename(file)
            voc.add_path(file)
            voc.add_database('Unknown')

            voc.add_size(H, W, C)
            voc.add_segmented('0')

            if points:
                for point in points:
                    voc.add_object()
                    voc.add_nptd(str(cls_name), 'Unspecified', '0', '0')
                    voc.add_bndbox(point[1], point[2], point[3], point[4])

            voc.build(os.path.join(save_path, file.replace('jpg', 'xml')))

def batch_demo_for_animal():
    txt_dir = r'C:\Users\1\Desktop\animals_txt'
    result_dir = r'C:\Users\1\Desktop\animals'
    xml_dir = r'C:\Users\1\Desktop\animals_xml'
    cls_list = os.listdir(txt_dir)
    result_list = os.listdir(result_dir)

    for cls in cls_list:
        txt_path = os.path.join(txt_dir, cls, "labels")
        result_cls_name = [i for i in result_list if i.startswith(cls)][0]
        result_path = os.path.join(result_dir, result_cls_name)
        xmls_path = os.path.join(xml_dir, result_cls_name)
        print(txt_path)
        print(result_path)
        print(xmls_path)
        print("=========================")
        demo_for_animal(result_path, txt_path, xmls_path, cls)



if __name__ == "__main__":
    batch_demo_for_animal()
