import os
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import csv
import argparse

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x) if isinstance(x, np.ndarray) else x.copy()
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

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

def Iou(boxA, boxB):
    if boxesIntersect(boxA, boxB) is False:
        return 0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA + 1) * (yB - yA + 1)

    area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #print(interArea / (area_A + area_B - interArea))
    iou = np.max([interArea/area_A, interArea/area_B])
    assert iou >= 0
    return iou

def Union_Box(boxA, boxB):
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])
    return(np.array([xA, yA, xB, yB]))

def Union_NMS(pred_box_list, iou_thres = 0.01):
    result_list = []
    if len(pred_box_list) > 1:
        #pred_box = sorted(pred_box_list, key=lambda s: (s[2] - s[0] + 1) * (s[3] - s[1] + 1))
        pred_box = pred_box_list
        idx = np.ones([len(pred_box)])
        while (np.sum(idx) > 0):
            ids = np.nonzero(idx)[0]
            union_box = pred_box[ids[0]]
            idx[ids[0]] = 0

            for i in ids[1:]:
                # print(ids[0], i)
                iou = Iou(union_box, pred_box[i])
                # print(iou)
                if iou > iou_thres:
                    idx[i] = 0
                    union_box = Union_Box(union_box, pred_box[i])

            result_list.append(union_box)
            # print("result_list:", result_list)
            # print("idx:", idx)
            if np.sum(idx) == 0:
                break
    else:
        result_list = pred_box_list
    return result_list

def draw_box(box_list, background= np.zeros([640, 640, 3]), anno = 'pred', color = (255, 0, 0)):
    for i, box in enumerate(box_list):
        cv2.rectangle(background, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color,
                      thickness=2)
        cv2.putText(background, anno + "_" + str(i), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

    # cv2.imshow("background", background)
    # cv2.waitKey(2000)

def get_sample():

    box1 = np.array([0.91875, 0.598148, 0.157292, 0.281481])
    box2 = np.array([0.386979, 0.796296, 0.130208, 0.187037])
    box3 = np.array([0.286979, 0.66296, 0.20208, 0.37037])
    box5 = np.array([0.347656, 0.406019, 0.695312, 0.284259])
    box6 = np.array([0.428906,0.423148,0.516146,0.638889])
    box_list = np.array([box1, box2, box3, box5, box6])
    return box_list

def batch_test_NMS():
    pred_result = glob.glob(os.path.join(r'C:\Users\1\Desktop\tmp\metric\labels' + "\*.txt"))
    for i, file in enumerate(pred_result[60:]):
        with open(file, 'r') as f:
            boxes = f.readlines()
        box_list = []
        if len(boxes) > 1:
            #print(boxes)
            for line_array in boxes:
                box = [float(i) for i in line_array.split()]
                box_list.append(np.array([box[1],box[2],box[3],box[4],]))
            box_list_ = xywhn2xyxy(np.array(box_list))
            print("{} {}\n{}\n".format(str(i),file,box_list_))
            draw_box(box_list_)
            y = Union_NMS(box_list_, 0.01)
            print(y)
            print("==========================================")
            draw_box(y)

def get_VOCbox(xml_file, class_name = 'smoke'):
    classes = ['smoke', 'building', 'greenhouse', 'road', 'raindrop', 'field', 'cloud', 'forest', 'lake',
               'lightspot', 'land', 'fog', 'snow', 'others', 'lightningrod']
    point_list = []
    if os.path.exists(xml_file):
        target = ET.parse(xml_file).getroot()
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            if name == class_name:
                bndbox = []
                bbox = obj.find('bndbox')
                pts = ['xmin', 'ymin', 'xmax', 'ymax']
                for t, pt in enumerate(pts):
                    cur_pt = int(bbox.find(pt).text) - 1
                    bndbox.append(cur_pt)
                point_list.append(bndbox)
    return point_list

def get_coco128box(txt_file, class_idx = 0, img_width = 1920, img_height = 1080, conf_thresh = 0.):
    xywh_box_list = []
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as f:
            files = f.readlines()
        for lines in files:
            line = [float(i) for i in lines.split()]
            cla = line[0]
            box = np.array([line[1], line[2], line[3], line[4]])
            conf = line[5] if len(line) == 6 else 1
            if cla == class_idx and conf > conf_thresh:
                xywh_box_list.append(box)
    #print(xywh_box_list)
    xyxy_box_list = xywhn2xyxy(np.array(xywh_box_list), img_width, img_height) if len(xywh_box_list) > 0 else xywh_box_list
    return xyxy_box_list

def get_FN_FP_nums(lab_box_list, pred_box_list):
    lab_nums = len(lab_box_list)
    pred_nums = len(pred_box_list)
    TP_in_lab_nums = 0
    TP_in_pred_nums = 0

    if lab_nums > 0 and pred_nums > 0:
        PR_metrix = np.zeros([lab_nums, pred_nums])
        for i, lab_box in enumerate(lab_box_list):
            for j, pred_box in enumerate(pred_box_list):
                iou = Iou(lab_box, pred_box)
                if iou > 0:
                    PR_metrix[i, j] = 1
        TP_in_lab_nums = np.sum(np.sum(PR_metrix, axis= 1) > 0)
        TP_in_pred_nums = np.sum(np.sum(PR_metrix, axis= 0) > 0)

    FP_in_pred_nums = pred_nums - TP_in_pred_nums
    FN_in_lab_nums = lab_nums - TP_in_lab_nums
    return lab_nums, pred_nums, FN_in_lab_nums, FP_in_pred_nums

class Metrics(object):
    def __init__(self, img_dir, label_dir, pred_dir, save_dir, save_img = False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.pred_dir = pred_dir
        self.img_path_list = os.listdir(img_dir)
        self.metric_matrix = np.zeros([len(self.img_path_list), 4])
        self.fp_path, self.fn_path, self.csv_path = self.mkdir_folder(save_dir)
        self.save_img = save_img

    def mkdir_folder(self, save_dir):
        fp_path = os.path.join(save_dir, 'fp_result')
        fn_path = os.path.join(save_dir, 'fn_result')
        if not os.path.exists(fp_path):
            os.makedirs(fp_path)
        if not os.path.exists(fn_path):
            os.makedirs(fn_path)
        csv_path = os.path.join(save_dir, os.path.basename(self.pred_dir) + ".csv")
        return fp_path, fn_path, csv_path

    def compute_recall_precision(self):

        for i, img_path in enumerate(self.img_path_list):

            lab_path = os.path.join(self.label_dir, img_path.split('.')[0] + ".xml")
            pred_path = os.path.join(self.pred_dir, img_path.split('.')[0] + ".txt")

            pred_boxes_ = get_coco128box(pred_path)
            lab_boxes_ = []
            smoke_boxes = get_VOCbox(lab_path, 'smoke')
            fog_boxes = get_VOCbox(lab_path, 'fog')
            lab_boxes_.extend(smoke_boxes)
            #lab_boxes_.extend(fog_boxes)

            lab_boxes = Union_NMS(lab_boxes_)
            pred_boxes = Union_NMS(pred_boxes_)
            print("lab_boxes", lab_boxes)
            print("pred_boxes", pred_boxes)
            self.metric_matrix[i, :] = get_FN_FP_nums(lab_boxes, pred_boxes)

            #deBug
            '''
            print("self.metric_matrix[i, :]:", self.metric_matrix[i, :])
            bg = cv2.imread(os.path.join(self.img_dir, img_path))
            draw_box(lab_boxes, bg, anno='lab', color=(255, 0, 0))
            draw_box(pred_boxes, bg, anno='pred', color=(0, 0, 255))
            m_bg = cv2.resize(bg, (960, 540))
            cv2.imshow("background", m_bg)
            cv2.waitKey(2000)
            '''

            if self.save_img:
                _, _, fn_num, fp_num = self.metric_matrix[i]
                if fp_num > 0 or fn_num > 0:
                    img = cv2.imread(os.path.join(self.img_dir, img_path))
                    draw_box(lab_boxes, background=img, anno='label', color=(255, 0, 0))
                    draw_box(pred_boxes, background=img, anno='pred', color=(0, 0, 255))
                    if fn_num > 0:
                        cv2.imwrite(os.path.join(self.fn_path, img_path) ,img)
                    if fp_num > 0:
                        cv2.imwrite(os.path.join(self.fp_path, img_path), img)

    def write_csv_file(self):
        t_lab, t_pred, t_fn, t_fp = np.sum(self.metric_matrix, axis=0)
        with open(self.csv_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "label_nums", "pred_nums", "fn_nums", "fp_nums"])
            for i, img_path in enumerate(self.img_path_list):
                r = [str(int(i)) for i in self.metric_matrix[i,:]]
                r.insert(0, img_path)
                writer.writerow(r)
            writer.writerow([" ", t_lab, t_pred, t_fn, t_fp, "recall:{}".format(1 - t_fn/t_lab), "precision:{}".format(1 - t_fp/t_pred)])

    def __str__(self):
        t_lab, t_pred, t_fn, t_fp = np.sum(self.metric_matrix, axis= 0)
        return "lab_total_nums:{}; pred_total_nums:{}\nFN_total_nums:{}; FP_total_nums:{}\nrecall:{}; precision:{}\n".format(t_lab, t_pred, t_fn, t_fp, 1 - t_fn/t_lab, 1 - t_fp/t_pred)

def run(img_dir, lab_dir, pred_dir, save_dir):
    me = Metrics(img_dir, lab_dir, pred_dir, save_dir, save_img=False)
    me.compute_recall_precision()
    print(me)
    me.write_csv_file()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default=r"E:\pycode\datasets\DZYH_test_dataset\yancut20220802_test_smoke\images", help='test img dir')
    parser.add_argument('--lab_dir', type=str, default=r"E:\pycode\datasets\DZYH_test_dataset\yancut20220802_test_smoke\xml", help='test label dir')
    parser.add_argument('--pred_dir', type=str, default=r"F:\Hi3559_box\result\yolov8x_pretrained_20230609_epoch40_confThre0.1", help='pred dir')
    parser.add_argument('--save_dir', type=str, default=r"F:\Hi3559_box\result\new_func", help='save dir')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)






