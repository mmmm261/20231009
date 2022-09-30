#生成与图像列表对应的空标签文件，txt或xml格式。
import os
import xml.etree.ElementTree as ET
def empty_cgwx_result(detect_result_path):
    for i in range(500):
        idx = str(i+1).zfill(6)
        txt_path = os.path.join(detect_result_path,"{}.txt".format(idx))
        with open(txt_path, 'w') as f:
            f.close()

def empty_tonghua_result(img_path,save_path):
    file_list = os.listdir(img_path)
    for file in file_list:
        txt_path = os.path.join(save_path, file.replace('jpg','xml'))
        with open(txt_path, 'a') as f:
            f.close()

def get_away_repeat(img_path,xml_path):
    img_file_list = [i.split('.')[0] for i in os.listdir(img_path)]
    xml_file_list = [i.split('.')[0] for i in os.listdir(xml_path)]
    unique_file_list = set(img_file_list + xml_file_list)
    for file in unique_file_list:
        xml_file = os.path.join(xml_path, "{}.xml".format(file))
        img_file = os.path.join(img_path, "{}.jpg".format(file))
        if os.path.exists(img_file) and os.path.exists(xml_file):
            os.remove(img_file)
            os.remove(xml_file)

def get_away_empty(img_path,xml_path):
    img_file_list = [i.split('.')[0] for i in os.listdir(img_path)]
    xml_file_list = [i.split('.')[0] for i in os.listdir(xml_path)]
    unique_file_list = set(img_file_list + xml_file_list)
    for file in unique_file_list:
        xml_file = os.path.join(xml_path, "{}.xml".format(file))
        img_file = os.path.join(img_path, "{}.jpg".format(file))
        print(xml_file)
        print(img_file)
        if not os.path.exists(xml_file):
            os.remove(img_file)
        elif not os.path.exists(img_file):
            os.remove(xml_file)
        else:
            with open(xml_file, 'r', encoding='utf-8') as f:
                target = f.readlines()
            if len(target)< 26:
                os.remove(xml_file)
                os.remove(img_file)


if __name__ == '__main__':
    detect_result_path = r'C:\Users\1\Desktop\detection-results'
    img_path = r'E:\pycode\datasets\DZYH_test_dataset\yan_cut20220802_test_smoke\images'
    xml_path = r'E:\pycode\datasets\DZYH_test_dataset\yan_cut20220802_test_smoke\xml'
    empty_cgwx_result(detect_result_path)