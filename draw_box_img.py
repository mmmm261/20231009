#Usage
#单帧烟雾结果可视化，在出现漏报与误报的图片上绘制检测框与真实标签

import csv
import cv2
import os

from get_metrics import get_coco128box, get_VOCbox

def draw_box(csv_path, label_path, pred_path, save_path):
    test_list = []

    with open(csv_path) as f :
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            if int(row[3]) + int(row[4]) > 0 :
                test_list.append(row[0])

    for test_file in test_list[:-1]:
        label_file = os.path.join(label_path + '\\' + test_file.split('\\')[-1].replace('jpg', 'xml'))
        pred_file = os.path.join(pred_path + '\\' + test_file.split('\\')[-1].replace('jpg', 'txt'))
        img = cv2.imread(test_file)
        print(test_file)
        H, W, _ = img.shape

        if os.path.exists(label_file):
            label_boxs = get_VOCbox(label_file)
            for label_point in label_boxs:
                cv2.rectangle(img, (label_point[0], label_point[1]), (label_point[2], label_point[3]), color = (255,0,0), thickness = 2)
                cv2.putText(img, 'label_smoke', (label_point[0], label_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 1)
        if os.path.exists(pred_file):
            pred_boxs = get_coco128box(pred_file, H, W)

            for pred_point in pred_boxs:
                cv2.rectangle(img, (pred_point[0], pred_point[1]), (pred_point[2], pred_point[3]), color = (0,0,255), thickness = 2)
                cv2.putText(img, 'pred_smoke', (pred_point[0], pred_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                #cv2.putText(img, 'smoke' , (pred_point[0], pred_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        new_path = os.path.join(save_path + '\\' + test_file.split('\\')[-1])
        cv2.imwrite(new_path, img)

if __name__ == '__main__':
    # yancut20220802_test_smoke
    # yzw20220704_test_no_smoke

    label_path = r'E:\pycode\datasets\DZYH_test_dataset\yancut20220802_test_smoke\xml'

    csv_path =  r'E:\pycode\yolov5-master_6.1\results\20220811_best_pretrained_20220822_best\yancut20220802_test_smoke.csv'
    pred_path = r'E:\pycode\yolov5-master_6.1\results\20220811_best_pretrained_20220822_best\yancut20220802_test_smoke\labels'

    save_path = r'E:\pycode\yolov5-master_6.1\results\20220811_best_pretrained_20220822_best\yancut20220802_test_smoke\lwb'

    # save_path = r'C:\Users\1\Desktop\metrics'
    # label_path = r'C:\Users\1\Desktop\metrics\labels'
    # pred_path = r'C:\Users\1\Desktop\metrics\pred'
    # csv_path = r'C:\Users\1\Desktop\metrics\1.csv'


    draw_box(csv_path, label_path, pred_path, save_path)