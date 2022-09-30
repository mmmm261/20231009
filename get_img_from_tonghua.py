#get_root_list：遍历文件夹下所有子文件夹，将带有xml与jpg、png的文件夹写入txt中
#get_img_label_from_root：遍历文件夹中的xml文件读出标签，标签为：0，1，2...，将不同标签的文件下的图片取出两张写入txt中
#get_notZoom 剔除Zoom文件夹
#smoke2to1 读写入单张图片

import os
import glob
import xml.etree.ElementTree as ET
import shutil
#img_lists = []

def get_root_list(file):
    with open(r'C:\Users\1\Desktop\tonghua_data.txt', 'w') as f:
        for root, dirs, files in os.walk(file):
            img_list = glob.glob(os.path.join(root + "\\*.jpg")) + glob.glob(os.path.join(root, "\\*.png"))
            xml_list = glob.glob(os.path.join(root + "\\*.xml"))
            if img_list :
                if xml_list:
                    print(root)
                    f.write(root)
                    f.write("\n")

def get_img_label_from_root(txt_path):
    with open(txt_path, 'r') as f:
        paths = f.readlines()
        paths = paths[85731:] #85731
    for path in paths:
        print(path)
        path = path.split('\n')[0]
        imgs = glob.glob(os.path.join(path + "\\*.jpg"))

        target_path = os.path.join(path, 'Fire.xml')
        if os.path.exists(target_path):
            targets = ET.parse(target_path).find('fireNum').text.lower().strip()

            if len(imgs) > 10:
                if targets == '0':
                    with open(r'C:\Users\1\Desktop\0.txt', 'a') as ff:
                        ff.write(os.path.join(path, imgs[-5]) + "\n")
                        ff.write(os.path.join(path, imgs[-10]) + "\n")
                elif targets == '1':
                    with open(r'C:\Users\1\Desktop\1.txt', 'a') as fff:
                        fff.write(os.path.join(path, imgs[-5]) + "\n")
                        fff.write(os.path.join(path, imgs[-10]) + "\n")
                elif targets == '2' or targets == '3':
                    with open(r'C:\Users\1\Desktop\2.txt', 'a') as fffff:
                        fffff.write(os.path.join(path, imgs[-5]) + "\n")
                        fffff.write(os.path.join(path, imgs[-10]) + "\n")
                else:
                    with open(r'C:\Users\1\Desktop\something_wrong.txt', 'a') as ffff:
                        print("危险危险")
                        ffff.write(path + "\n")

        else:
            with open(r'C:\Users\1\Desktop\something_wrong.txt', 'a') as ffff:
                print("危险危险")
                ffff.write(path + "\n")

def get_notZoom(txt_path):
    with open(txt_path, 'r') as f:
        files = f.read().splitlines()
    for file in files:
        if not file.split('\\')[-2].endswith('Zoomsmoke'):
            with open(r'C:\Users\1\Desktop\0_smoke.txt', 'a') as ff:
                ff.write(file + '\n')
        else:
            with open(r'C:\Users\1\Desktop\0_Zoom.txt', 'a') as fff:
                fff.write(file + '\n')

def smoke2to1(txt_path):
    with open(txt_path, 'r') as f:
        files = f.read().splitlines()
    for i, file in enumerate(files):
        if i % 2 == 0 :
            with open(r'C:\Users\1\Desktop\0_smoke_1.txt', 'a') as ff:
                ff.write(file + '\n')

def get_img_label_from_local(file_dir, txt_path):
    file_list = os.listdir(file_dir)
    for file in file_list:
        file_path = os.path.join(file_dir, file)
        img_path = os.path.join(file_path, 'f1.jpg')
        if os.path.exists(img_path):
            with open(txt_path, 'a') as f:
                f.write(img_path + '\n')

def process_ce_data(data_path):
    txt_path = os.path.join(r'Z:\2022_Data\test_data\ce_processed_data', "{}.txt".format(data_path.split('\\')[-1]))
    file_list = ['images', 'txt', 'xmls', 'result']
    with open(txt_path, 'a') as f:
        for root, dirs, files in os.walk(data_path):
            if not dirs:
                if files:
                    file = [i for i in files if i.endswith("25.jpg") or i.endswith("40.jpg")]

                    if file:
                        f.write(os.path.join(root, file[-1]) + '\n')
                    else:
                        if [i for i in files if i.endswith(".jpg")]:
                            f.write(os.path.join(root, [i for i in files if i.endswith(".jpg")][-1]) + '\n')
    if not os.path.exists(txt_path.split('.')[0]):
        os.mkdir(txt_path.split('.')[0])
    for file_name in file_list:
        if not os.path.exists(os.path.join(txt_path.split('.')[0],file_name)):
            os.mkdir(os.path.join(txt_path.split('.')[0], file_name))

def get_data_from_ce(img_path, xml_path):

    img_list = [os.path.join(img_path, i.replace('xml', 'jpg'))for i in os.listdir(xml_path)]
    if not os.path.exists(img_path.replace('images', 'split_img')):
        os.mkdir(img_path.replace('images', 'split_img'))

    for img_path in img_list:
        shutil.copy(img_path, img_path.replace('images', 'split_img'))

def remove_img_from_empty_txt(img_path, txt_path):
    for txt in os.listdir(txt_path):
        with open(os.path.join(txt_path, txt), 'r') as f:
            points = f.readlines()
        if not points:
            #print(os.path.join(img_path, txt.replace('txt', 'jpg')))
            #print(os.path.join(txt_path, txt))
           os.remove(os.path.join(img_path, txt.replace('txt', 'jpg')))
           os.remove(os.path.join(txt_path, txt))


if __name__ == '__main__':
    img_path = r'E:\pycode\datasets\DZYH_train_dataset\ce20220822\images'
    txt_path = r'E:\pycode\datasets\DZYH_train_dataset\ce20220822\txt'
    remove_img_from_empty_txt(img_path, txt_path)