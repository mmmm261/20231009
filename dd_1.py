import cv2
import numpy as np
#
# mask1_path = r'E:\pycode\datasets\GID_dataset\Large-scale Classification_5classes\label_5classes_png\GF2_PMS2__L1A0001757484-MSS2_label.png'
# mask2_path = r'E:\pycode\Remote_sensing_segmentation\run\test_result\GF2_PMS2__L1A0001757484-MSS2_result.png'
# mask1 = cv2.imread(mask1_path, 0)
# mask2 = cv2.imread(mask2_path, 0)
# print(np.unique(mask1))
# print(np.unique(mask2))

background = np.zeros([600,600])
background[:100,:] = 0
background[100:200,:] = 51
background[200:300,:] = 102
background[300:400,:] = 153
background[400:500,:] = 204
background[500:,:] = 255
cv2.imwrite(r'C:\Users\1\Desktop\label_cla\background.png', background)
