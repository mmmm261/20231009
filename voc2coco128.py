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
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    classes = ['smoke', 'building', 'greenhouse', 'road', 'raindrop', 'field', 'cloud', 'forest', 'lake', 'lightspot', 'land', 'fog', 'snow', 'others', 'lightningrod']
    imgs = sorted(glob.glob(os.path.join(img_path + '\\*.jpg')) + glob.glob(os.path.join(img_path + '\\*.png')))
    xmls = sorted(glob.glob(os.path.join(xml_path + '\\*.xml')))

    for p in range(len(imgs)):
        img = cv2.imread(imgs[p])
        h, w, _ = img.shape
        xml_file = os.path.join(xml_path, "{}.xml".format(os.path.basename(imgs[p]).split('.')[0]))
        print(imgs[p])
        print(xml_file)
        lists = []
        print(lists)
        if os.path.exists(xml_file):
            target = ET.parse(xml_file).getroot()
            for obj in target.iter('object'):
                name = obj.find('name').text.lower().strip()
                if name == 'fog':
                    #idx = 0 if classes.index(name) == 0 else 1
                    idx = 0
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



        new_txt_path = os.path.join(txt_path, "{}.txt".format(os.path.basename(imgs[p]).split('.')[0]))
        print(new_txt_path)
        print(lists)
        with open(new_txt_path,'w') as f :
            for i in range(len(lists)):
                for j in range(5):
                    f.write(str(lists[i][j])+" ")
                f.write("\n")



def batch_voc2coco():
    for p in os.listdir(r'F:\tmp_single_frame_data_tagging\tmp_single_frame_data_after_cls'):
        img_path = r"F:\tmp_single_frame_data_tagging\tmp_single_frame_data_after_cls\{}\images".format(p)
        xml_path = r"F:\tmp_single_frame_data_tagging\tmp_single_frame_data_after_cls\{}\xmls".format(p)
        txt_path = r"F:\tmp_single_frame_data_tagging\tmp_single_frame_data_after_cls\{}\labels".format(p)
        voc2coco(img_path, xml_path, txt_path)

if __name__ == '__main__':
    # batch_voc2coco()
    img_path = r'Z:\qdm\fog_detection\20230828\images'
    xml_path = r'Z:\qdm\fog_detection\20230828\xmls'
    txt_path = r'Z:\qdm\fog_detection\20230828\labels'
    voc2coco(img_path, xml_path, txt_path)