import numpy as np
import cv2
import os
import time
import csv
import argparse

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / (union + 1)
        mIoU = np.nanmean(IoU)
        return IoU, mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)

        label = self.numClass * imgLabel[mask] + imgPredict[mask]

        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def run(pred_path, label_path, csv_save_path):
    pred_list = os.listdir(pred_path)
    label_list = os.listdir(label_path)
    iou_result = []
    use_t = 0
    metric = SegmentationMetric(numClass = 6)
    for i in range(len(pred_list)):

        pred_file = np.array(cv2.imread(os.path.join(pred_path, pred_list[i]), 0) / 51, dtype= int)
        label_file = np.array(cv2.imread(os.path.join(label_path, label_list[i]), 0) / 51, dtype= int)

        dt1 = time.time()
        metric.confusionMatrix = metric.genConfusionMatrix(pred_file, label_file)
        iou, miou = metric.meanIntersectionOverUnion()
        dt2 = time.time()
        use_t += dt2 - dt1

        iou = list(iou)
        iou.append(miou)
        iou = [round(i, 3) for i in iou]
        iou.insert(0, os.path.basename(pred_list[i]))
        iou_result.append(iou)

    print(use_t, "s")

    with open(csv_save_path, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "background", "water", "grass", "tree", "farmland", "build_up", "miou"])
        for r in iou_result:
            writer.writerow(r)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default = r'E:\pycode\Remote_sensing_segmentation\run\resnest50_pretrained_DANet_20221026_epoch25')
    parser.add_argument('--label_path', type=str, default = r'E:\pycode\datasets\GID_dataset\Large-scale Classification_5classes\label_5classes_png')
    parser.add_argument('--csv_save_path', type=str, default=r'E:\pycode\Remote_sensing_segmentation\run\resnest50_pretrained_DANet_20221026_epoch25.csv')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))