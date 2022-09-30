# Usage
#VOC的xml格式数据集，转化为coco128格式，直接用于yolov5的训练。

# img_path = r'C:\Users\QDM\Desktop\wrj\0304'
# xml_path = r'C:\Users\QDM\Desktop\wrj\0304\outputs'
# txt_path = r'C:\Users\QDM\Desktop\wrj\0304\txt'
# voc2coco(img_path,xml_path,txt_path)
##注意：写入类别
import cv2
import xml.etree.ElementTree as ET
import glob
import os

def voc2coco(img_path,xml_path,txt_path):
    classes = ['smoke', 'building', 'greenhouse', 'road', 'raindrop', 'field', 'cloud', 'forest', 'lake', 'lightspot', 'land', 'fog', 'snow', 'others', 'lightningrod']
    imgs = glob.glob(os.path.join(img_path + '\\*.jpg'))
    xmls = glob.glob(os.path.join(xml_path + '\\*.xml'))

    for p in range(len(imgs)):
        print(imgs[p])
        print(xmls[p])
        img = cv2.imread(imgs[p])
        h, w, _ = img.shape
        target = ET.parse(xmls[p]).getroot()
        lists = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            if name != 'fog':
                idx = 0 if classes.index(name) == 0 else 1
                bbox = obj.find('bndbox')
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                bndbox = []
                try:
                    for t, pt in enumerate(pts):
                        cur_pt = int(bbox.find(pt).text) - 1
                        bndbox.append(cur_pt)
                    centerx = round(((bndbox[2] - bndbox[0]) / 2 + bndbox[0])/w,6)
                    centery = round(((bndbox[3] - bndbox[1]) / 2 + bndbox[1])/h,6)
                    ww = round((bndbox[2] - bndbox[0]) / w,6)
                    hh = round((bndbox[3] - bndbox[1])/ h,6)
                    lists.append([idx, centerx, centery, ww, hh])
                except:
                    pass

        new_txt_path = os.path.join(txt_path, xmls[p].split('\\')[-1].replace('xml', 'txt'))
        with open(new_txt_path,'w') as f :
            for i in range(len(lists)):
                for j in range(5):
                    f.write(str(lists[i][j])+" ")
                f.write("\n")
if __name__ == '__main__':

    img_path = r'E:\pycode\datasets\DZYH_train_dataset\ce20220822\images'
    xml_path = r'E:\pycode\datasets\DZYH_train_dataset\ce20220822\xmls'
    txt_path = r'E:\pycode\datasets\DZYH_train_dataset\ce20220822\txt'
    voc2coco(img_path, xml_path, txt_path)