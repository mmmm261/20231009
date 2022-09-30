#Usage get_metric 、draw_box_img的草纸

#labels: VOC voc2coco128.py txt
#results: txt

## create_txt.py
## copy results, labels to empty

##coco128_voc_txt.py labels,results

import glob
import os
import xml.etree.ElementTree as ET
import shutil
from shutil import copyfile

def create_empty_txt(metric_path,test_path):
    result_path = os.path.join(metric_path,'result_path')
    label_path = os.path.join(metric_path,'label_path')

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    file_list = glob.glob(os.path.join(test_path + '\\*.jpg'))

    for file in file_list:
        idx = file.split('\\')[-1].replace('jpg', 'txt')
        path1 = os.path.join(result_path + "\\" + idx )
        path2 = os.path.join(label_path + "\\" +idx)

        with open(path1, 'w') as f :
            f.close()
        with open(path2, 'w') as f:
            f.close()

# def my_copy(result_path,empty_path):
#     result_list = glob.glob(result_path + '\\*.txt')
#     for file in result_list:
#         print(file)
#         shutil.copyfile(file,empty_path)

#labels: VOC.xml return VOC(txt)
def trans_label(xml_path, txt_path):
    xmls = glob.glob(os.path.join(xml_path + '\*.xml'))

    for p in range(len(xmls)):
        target = ET.parse(xmls[p]).getroot()
        lists = []
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for t, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)

            lists.append([bndbox[0], bndbox[1], bndbox[2], bndbox[3]])

        new_txt_path = os.path.join(txt_path, xmls[p].split('\\')[-1].replace('xml' , 'txt'))
        print(new_txt_path)
        with open(new_txt_path, 'w') as f:
            for i in range(len(lists)):
                f.write(str(0) + " ")
                for j in range(4):
                    f.write(str(lists[i][j]) + " ")
                f.write("\n")
#preds: coco128.txt return VOC(txt)
def trans_pred(txt_path, new_txt_path,H,W):
    txt_file = glob.glob(os.path.join(txt_path + '\\*.txt'))

    for file in txt_file:
        print(file)
        with open (file ,'r') as f:
            files = f.readlines()
        point_list = []
        if len(files) > 0 :
            for lines in files:
                line = [float(i) for i in lines.split()]
                _,x,y,h,w,conf = line
                xx = x * W
                yy = y * H
                hh = h * H
                ww = w * W
                min_left = xx - ww / 2
                min_top = yy - hh / 2
                max_right = xx + ww / 2
                max_down = yy + hh / 2
                point_list.append([min_left,min_top,max_right,max_down,conf])
        else:
            f.close()

        nnew_txt_path = os.path.join(new_txt_path, file.split('\\')[-1])
        with open(nnew_txt_path, 'w') as f:
            for i in range(len(point_list)):
                f.write(str(0) + " ")
                for j in range(5):
                    f.write(str(point_list[i][j]) + " ")
                f.write("\n")


if __name__ == '__main__':
    metric_path = r'E:\Program Files\feiq\Recv Files\metric_path'
    test_path = r'E:\Program Files\feiq\Recv Files\202205\202205'
    #create_empty_txt(metric_path, test_path)
    result_path = r'E:\Program Files\feiq\Recv Files\DZYH_results'
    empty_path = r'E:\Program Files\feiq\Recv Files\metric_path\result_path'

    xml_path = r'E:\Program Files\feiq\Recv Files\202205\xml'
    txt_path = r'E:\Program Files\feiq\Recv Files\metric_path\label_path'
    #trans_label(xml_path, txt_path)
    trans_pred(result_path,empty_path, 1080, 1920)

