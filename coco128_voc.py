#yolov5的txt检测结果 转换为xml格式，用于斯总的数据标注，进行下一轮算法迭代。
import os
import cv2
from xml.dom import minidom
import numpy as np

class VOC_Sample_Generator:

    def __init__(self):
        self.dom = minidom.Document()
        self.root_node = self.dom.createElement('annotation')
        self.object_node = self.dom.createElement('object')
        self.root_node.appendChild(self.object_node)

    # def add_object(self):
    #     self.object_node = self.dom.createElement('object')
    #     self.root_node.appendChild(self.object_node)

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

    return cv2.imdecode(np.fromfile(img_path, np.uint8), flags = cv2.IMREAD_COLOR).shape


def get_xyxy(txt_path, W, H):
    with open(txt_path, 'r') as f:
        files = f.readlines()
    point_list = []
    for lines in files:
        line = [float(i) for i in lines.split()]
        cla, x, y, w, h, conf = line
        xx = x * W
        yy = y * H
        hh = h * H
        ww = w * W
        min_left = int(xx - ww / 2)
        min_top = int(yy - hh / 2)
        max_right = int(xx + ww / 2)
        max_down = int(yy + hh / 2)
        point_list.append([cla, min_left, min_top, max_right, max_down])
    return point_list
# 测试
if __name__ == "__main__":
    result_path = r'Z:\2022_Data\test_data\ce_processed_data\ce20220819\result'
    txt_path = r'Z:\2022_Data\test_data\ce_processed_data\ce20220819\txt'
    save_path = r'Z:\2022_Data\test_data\ce_processed_data\ce20220819\xmls'

    file_list = os.listdir(result_path)#os.path.join(result_path + '\\*.jpg')
    for file in file_list:
        print(file)
        if file.endswith('jpg'):
            points = []
            W,H,C = get_HWC(os.path.join(result_path, file))
            if os.path.exists(os.path.join(txt_path,file.replace('jpg','txt'))):
            #if os.path.exists(os.path.join(txt_path,"{}.txt".format(file))):
                points = get_xyxy(os.path.join(txt_path, file.replace('jpg','txt')), H, W)
            voc = VOC_Sample_Generator()


            voc.add_folder(r'images')
            voc.add_filename(file)
            voc.add_path(os.path.join(result_path, file))
            voc.add_database('Unknown')

            voc.add_size(H, W, C)
            voc.add_segmented('0')

            voc.add_nptd('smoke','Unspecified','0','0')

            if points:
                for point in points:
                    if point[0] == 0:
                        voc.add_bndbox(point[1], point[2], point[3], point[4])

            voc.build(os.path.join(save_path ,file.replace('jpg','xml')))

