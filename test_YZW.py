#Usage
#查找图片文件，读取标签，将结果保存至本地。
import os
import xml.etree.ElementTree as ET
import glob
import cv2
#xml = r'Z:\wangjinlong\presetModeData\080506\savedData-11-080506\savedData\2021_08_05_08_01_09_deviceId11_presetId205_smoke\Fire.xml'



base_path = r'Z:\wangjinlong\presetModeData'
# date_path1 = os.listdir(base_path)[:48] + os.listdir(base_path)[51:84] + [os.listdir(base_path)[89],os.listdir(base_path)[-12]]
# date_path2 = [os.listdir(base_path)[49], os.listdir(base_path)[50]] + os.listdir(base_path)[85:88]

date_path1 = os.listdir(base_path)[:48] + os.listdir(base_path)[51:84]
device_list1 = []

for date_path in date_path1:
    savedData_paths = os.listdir(os.path.join(base_path,date_path))
    for savedData_path in savedData_paths:
        s_paths = os.path.join(base_path,date_path,savedData_path)
        save_paths = os.listdir(s_paths)
        for save_path in save_paths:
            deviced_path = os.path.join(s_paths,save_path)
            device_list1.append(deviced_path)
print(len(device_list1))

smoke_list = []
for path in device_list1:
    #print(os.path.join(path, os.listdir(path)[0]))
    smoke_list.append(os.path.join(path, os.listdir(path)[0]))
#print(len(smoke_list))

img_paths = []
i = 0
for smoke_path in smoke_list:
    img_path = glob.glob(os.path.join(smoke_path + '\\*.jpg'))
    if img_path:
        targets = ET.parse(os.path.join(smoke_path , 'Fire.xml')).find('fireNum').text.lower().strip()
        array = cv2.imread(img_path[2])
        new_path = os.path.join(r'E:\pycode\datasets\YZW_yan',str(targets),"{}.jpg".format(i))
        cv2.imwrite(new_path, array)
    i = i + 1

