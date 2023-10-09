#Usage
#test_path get
#labels_path get label_box VOC
#preds_path get pred_box coco128

#统计单帧烟雾检测的误报与漏报

import os
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import csv
import itertools

def get_VOCbox(xml_file):
    target = ET.parse(xml_file).getroot()
    classes = ['smoke', 'building', 'greenhouse', 'road', 'raindrop', 'field', 'cloud', 'forest', 'lake',
               'lightspot', 'land', 'fog', 'snow', 'others', 'lightningrod']
    point_list = []
    for obj in target.iter('object'):
        name = obj.find('name').text.lower().strip()
        if name == 'smoke':
            bndbox = []
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for t, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            point_list.append(bndbox)
    return point_list

def get_coco128box(txt_file, H, W):
    with open(txt_file, 'r') as f:
        files = f.readlines()
    point_list = []
    for lines in files:
        line = [float(i) for i in lines.split()]
        if len(line) > 5:
            cla, x, y, w, h, conf = line
        else:
            cla, x, y, w, h = line
        if cla == 0 :
        #if cla > 0 :
            xx = x * W
            yy = y * H
            hh = h * H
            ww = w * W

            # center xy, w, h
            min_left = int(xx - ww / 2)
            min_top = int(yy - hh / 2)
            max_right = int(xx + ww / 2)
            max_down = int(yy + hh / 2)

            # left_up xy, w, h
            # min_left = int(xx)
            # min_top = int(yy)
            # max_right = int(xx + ww)
            # max_down = int(yy + hh)

            if len(line) > 5:
                point_list.append([min_left, min_top, max_right, max_down, conf])
            else:
                point_list.append([min_left, min_top, max_right, max_down])

    return point_list

def boxesIntersect(boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxB[0] > boxA[2]:
        return False  # boxA is left of boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    return True

def get_iou(boxA, boxB):
    if boxesIntersect(boxA, boxB) is False:
        return 0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)

    area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = np.max([interArea/area_A, interArea/area_B])
    assert iou >= 0
    return iou

def get_metrics(label_box, pred_box, iou_thres):
    TL = 0
    idx = []
    total_label = len(label_box)
    total_pred = len(pred_box)

    for l_box in label_box :
        iou_list = []
        for p_box in pred_box:
            Iou = get_iou(l_box, p_box)
            iou_list.append(Iou)
        if np.max(iou_list) > iou_thres:
            TL = TL + 1
        for iou in iou_list :
            if iou > iou_thres:
                idx.append(iou_list.index(iou))
    TP = len(np.unique(idx))
    return total_label, total_pred, TL, TP

def pred_box_NMS(pred_box):
    if len(pred_box) > 1:
        pred_box = sorted(pred_box, key = lambda s:(s[2] - s[0] + 1) * (s[3] - s[1] + 1))
        combination_list = list(itertools.combinations(pred_box, 2))
        for combi_list in combination_list:
            iiou = get_iou(combi_list[0], combi_list[1])
            if iiou == 1:
                try:
                    pred_box.remove(combi_list[0])
                except:
                    pass
    return pred_box

def get_metrices(test_path, label_path, pred_path, iou_thres):
    test_list = glob.glob(test_path + '\\*.jpg')
    total_list = []
    for test_file in test_list :
        label_file = os.path.join(label_path + '\\' + test_file.split('\\')[-1].replace('jpg','xml'))
        tmp_name = test_file.split('\\')[-1].split('.')[0]
        pred_file = os.path.join(pred_path + '\\' + tmp_name + '.txt')
        #pred_file = os.path.join(pred_path + '\\' + test_file.split('\\')[-1].replace('jpg','txt'))
        print(label_file)
        print(pred_file)

        H, W, _ = cv2.imread(test_file).shape
        lab_amount = 0
        pred_amount = 0
        if os.path.exists(label_file):
            label_boxs = get_VOCbox(label_file)
            lab_amount = len(label_boxs)
        if os.path.exists(pred_file):
            pred_box = get_coco128box(pred_file, H ,W)
            pred_boxs = pred_box_NMS(pred_box)

            pred_amount = len(pred_boxs)
        if lab_amount != 0 and pred_amount != 0 :
            total_label, total_pred, TL, TP = get_metrics(label_boxs, pred_boxs, iou_thres)

            LB = total_label - TL
            WB = total_pred - TP
        else:
            total_label = lab_amount
            total_pred = pred_amount
            if lab_amount > pred_amount:
                LB = lab_amount
                WB = 0
            elif lab_amount == pred_amount:
                LB = 0
                WB = 0
            else:
                LB = 0
                WB = pred_amount
        total_list.append([test_file, total_label, total_pred, LB, WB])
    return total_list

def write_result_csv(total_list, csv_path):
    total_objs = np.sum([i[1] for i in total_list])
    total_preds = np.sum([i[2] for i in total_list])
    total_LB = np.sum([i[3] for i in total_list])
    total_WB = np.sum([i[4] for i in total_list])

    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "total_label", "total_pred", "LB", "WB"])
        for r in total_list:
            writer.writerow(r)
        writer.writerow([" ", str(total_objs), str(total_preds), str(total_LB), str(total_WB)])

def test_standart_data():
    test_smoke = True
    if test_smoke:
        test_path = r'E:\pycode\datasets\DZYH_test_dataset\yancut20220802_test_smoke\images'
        label_path = r'E:\pycode\datasets\DZYH_test_dataset\yancut20220802_test_smoke\xml'
    else:
        test_path = r'E:\pycode\datasets\DZYH_test_dataset\yzw20220704_test_no_smoke\images'
        label_path = r'E:\pycode\datasets\DZYH_test_dataset\yzw20220704_test_no_smoke\xml'

    # pred_path = r'E:\pycode\yolov5-master_6.1\runs\detect\epoch50_pretrained_20220811_best_3559_pc_result\labels'
    # csv_path = r'E:\pycode\yolov5-master_6.1\runs\detect\epoch50_pretrained_20220811_best_3559_pc_result.csv'
    pred_path = r'F:\Hi3559_box\result\txt_result_confThresh0.15_modelrgb_databgr'
    csv_path = r'F:\Hi3559_box\result\txt_result_confThresh0.15_modelrgb_databgr.csv'

    total_list = get_metrices(test_path, label_path, pred_path, 0.1)

    write_result_csv(total_list, csv_path)

def batch_test():
    total_list = []

    for p in os.listdir(r'F:\tmp_single_frame_data_tagging')[1:]:
        print(p)
        img_path = r"F:\tmp_single_frame_data_tagging\{}\images".format(p)
        lab_path = r"F:\tmp_single_frame_data_tagging\{}\xmls".format(p)
        pred_path = r"F:\tmp_single_frame_data_tagging\{}\20220811_best_pretrained_20220830_best_result_txt".format(p)
        total_list += get_metrices(img_path, lab_path, pred_path, 0.1)

    csv_path = r'C:\Users\1\Desktop\tmp\20220811_best_pretrained_20220830_20230309_result_.csv'
    write_result_csv(total_list, csv_path)

def dd():
    origal_txt_path = os.listdir(r'E:\pycode\datasets\DZYH_test_dataset\yancut20220802_test_smoke\txt')
    for txt_name in origal_txt_path:
        new_txt_name = txt_name.split('.')[0]
        print(new_txt_name)
        print(new_txt_name[:-1])
        #new_txt = new_txt_name[]

if __name__ == '__main__':
    #dd()
    test_standart_data()