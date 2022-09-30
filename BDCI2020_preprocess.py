from PIL import Image
import numpy as np
import os

label_path = r'E:\pycode\datasets\BDCI2020-seg_dataset\origal_dataset\lab_train'
building_path = [r'E:\pycode\datasets\BDCI2020-seg_dataset\building_dataset\building_labels', r'E:\pycode\datasets\BDCI2020-seg_dataset\building_dataset\no_building_labels']
arable_land_path = [r'E:\pycode\datasets\BDCI2020-seg_dataset\building_dataset\arable_land_labels', r'E:\pycode\datasets\BDCI2020-seg_dataset\building_dataset\no_arable_land_labels']
label_list = [os.path.join(label_path, i) for i in os.listdir(label_path)]
for label_file in label_list:
    lab = np.asarray(Image.open(label_file), dtype= 'uint16')
    lab[lab == 0] = 256
    lab[lab < 256] = 0
    if np.unique(lab).any() :
        Image.fromarray(lab).convert('RGB').save(os.path.join(building_path[0], label_file.split('\\')[-1]))
    else:
        Image.fromarray(lab).convert('RGB').save(os.path.join(building_path[1], label_file.split('\\')[-1]))

for label_file in label_list:
    lab = np.asarray(Image.open(label_file), dtype= 'uint16')
    lab[lab == 1] = 256
    lab[lab < 256] = 0
    if np.unique(lab).any() :
        Image.fromarray(lab).convert('RGB').save(os.path.join(arable_land_path[0], label_file.split('\\')[-1]))
    else:
        Image.fromarray(lab).convert('RGB').save(os.path.join(arable_land_path[1], label_file.split('\\')[-1]))
