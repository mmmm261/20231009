#============================多文件夹xml文件转json============================#
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json

class_names = ["drone"]  # 本人转换的标签只有1个
img_id = -1  # 全局变量
anno_id = -1  # 全局变量


def voc2coco(data_dir, dir_save, jsonname):  # 获取所有的xml文件列表
    xml_dir = data_dir
    # 多个文件夹操作
    train_xmls = list()
    for tt in tqdm(os.listdir(xml_dir)):  # 遍历文件夹获取xml文件列表
        one_dir_path = os.path.join(xml_dir, tt)
        train_xmls = train_xmls + [os.path.join(one_dir_path, n) for n in os.listdir(one_dir_path)]

    ## 单个文件夹操作
    # train_xmls = [os.path.join(xml_dir, n) for n in os.listdir(xml_dir)]

    print('got xmls')

    train_coco = xml2coco(train_xmls)
    with open(os.path.join(dir_save, jsonname), 'w') as f:  # 保存进入指定的coco文件夹
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    print('done')


def xml2coco(xmls):  # 将多个xml文件转换成coco格式
    coco_anno = {'info': {"description": "yip_make"}, 'images': [], 'licenses': [], 'annotations': [], 'categories': []}
    coco_anno['categories'] = [{'supercategory': "", 'id': i, 'name': j} for i, j in enumerate(class_names)]
    global img_id, anno_id
    for fxml in tqdm(xmls):  # 逐一对xml文件进行处理
        try:
            tree = ET.parse(fxml)
            objects = tree.findall('object')
        except:
            print('err xml file: ', fxml)
            continue
        img_id += 1  # 无论图片有无标签，该图片也算img_id
        size = tree.find('size')
        ih = int(size.find('height').text)
        iw = int(size.find('width').text)
        img_name = fxml.replace("Annotations", "Data").replace("xml", "JPEG")  # 获得原始图片的路径
        img_info = {}
        img_info['id'] = img_id
        img_info['file_name'] = img_name
        img_info['height'] = ih
        img_info['width'] = iw
        coco_anno['images'].append(img_info)
        if len(objects) < 1:
            print('no object in ', fxml)  # 打印没有bndbox标签的xml文件
            continue
        for obj in objects:  # 获取xml内的所有bndbox标签
            cls_name = obj.find('name').text
            bbox = obj.find('bndbox')
            x1 = int(bbox.find('xmin').text)
            y1 = int(bbox.find('ymin').text)
            x2 = int(bbox.find('xmax').text)
            y2 = int(bbox.find('ymax').text)
            if x2 < x1 or y2 < y1:
                print('bbox not valid: ', fxml)
                continue
            anno_id += 1
            bb = [x1, y1, x2 - x1, y2 - y1]
            categery_id = class_names.index(cls_name)
            area = (x2 - x1) * (y2 - y1)
            anno_info = {}
            anno_info['segmentation'] = []
            anno_info['area'] = area
            anno_info['image_id'] = img_id
            anno_info['bbox'] = bb
            anno_info['iscrowd'] = 0
            anno_info['category_id'] = categery_id
            anno_info['id'] = anno_id
            coco_anno['annotations'].append(anno_info)

    return coco_anno


if __name__ == '__main__':
    ## 单个文件夹
    # data_dir = 'ILSVRC2015/Annotations/VID/train/01_1667_0001-1500'

    # data_dir含多个文件夹，每个文件夹有多个xml文件
    data_dir = 'ILSVRC2015/Annotations/VID/val/'  # VID格式xml训练集或测试集的文件夹

    dir_save = 'coco/annotations'  # coco文件保存的文件夹
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    # jsonname = 'vid_train_coco.json'  # coco文件名
    jsonname = 'vid_val10000_coco.json'  # coco文件名
    voc2coco(data_dir, dir_save, jsonname)

