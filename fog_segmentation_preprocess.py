import cv2
import os

def seg_src_mask(img_path, mask_path, save_path):
    image_save_path = os.path.join(save_path, 'images')
    label_save_path = os.path.join(save_path, 'labels')
    img_list = os.listdir(img_path)
    for file in img_list:
        img = cv2.imread(os.path.join(img_path, file))
        mask = cv2.imread(os.path.join(mask_path, file.replace('jpg','png')))
        k = 0
        for i in range(4):
            for j in range(2):
                img_ = img[j * 540 : j * 540 + 540, i * 460:i * 460 + 540, :]
                mask_ = mask[j * 540:j * 540 + 540, i * 460:i * 460 + 540, 2]

                img_name = "{}.jpg".format(file.split('.')[0] + '_' + str(k))
                mask_name = "{}.png".format(file.split('.')[0] + '_' + str(k))
                k += 1
                print(os.path.join(image_save_path, img_name))
                print(os.path.join(label_save_path, mask_name))
                cv2.imwrite(os.path.join(image_save_path, img_name), img_)
                cv2.imwrite(os.path.join(label_save_path, mask_name), mask_)

if __name__ == '__main__':
    img_path = r'E:\pycode\datasets\fog_segmentation\src'
    mask_path = r'E:\pycode\datasets\fog_segmentation\mask'
    save_path = r'E:\pycode\datasets\fog_segmentation\fog_segmentation_20220826'
    seg_src_mask(img_path, mask_path, save_path)