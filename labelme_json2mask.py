# 调用labelme库中原有的 labelme_json_to_dataset 为核心
# 批量将文件夹中的json文件转换，并抽取对应图片至各自文件夹

import os
import shutil
import argparse


def GetArgs():
    parser = argparse.ArgumentParser(description='将labelme标注后的json文件批量转换为图片')
    parser.add_argument('--input', '-i', default = r'E:\pycode\datasets\cgwx_20220920\JL1GF02B_PMS2_20220711092631_200092286_102_0016_001_L3D_PSH_206212\labelme_json_result', help='json文件目录')
    parser.add_argument('--out-mask', '-m',  default = r'E:\pycode\datasets\cgwx_20220920\JL1GF02B_PMS2_20220711092631_200092286_102_0016_001_L3D_PSH_206212\labelme_mask_result', help='mask图存储目录')
    parser.add_argument('--out-img', '-r', help='json文件中提取出的原图存储目录')
    parser.add_argument('--out-viz', '-v', default = r'E:\pycode\datasets\cgwx_20220920\JL1GF02B_PMS2_20220711092631_200092286_102_0016_001_L3D_PSH_206212\labelme_maskupimg',help='mask与原图合并viz图存储目录')
    return parser.parse_args()


if __name__ == '__main__':
    _args = GetArgs()
    _jsonFolder = _args.input

    input_files = os.listdir(_jsonFolder)
    for sfn in input_files:                                     # single file name
        if (os.path.splitext(sfn)[1] == ".json"):               # 是否为json文件

            try:
            # 调用labelme_json_to_dataset执行转换,输出到 temp 文件夹
                os.system("labelme_json_to_dataset %s -o temp" % (_jsonFolder + '/' + sfn))

                # 复制json文件中提取出的原图到存储目录
                if _args.out_img:
                    if not os.path.exists(_args.out_img):           # 文件夹是否存在
                        os.makedirs(_args.out_img)

                    src_img = "temp\img.png"
                    dst_img = _args.out_img + '/' + os.path.splitext(sfn)[0] + ".png"
                    shutil.copyfile(src_img, dst_img)

                # 复制mask图到存储目录
                if _args.out_mask:
                    if not os.path.exists(_args.out_mask):          # 文件夹是否存在
                        os.makedirs(_args.out_mask)

                    src_mask = "temp\label.png"
                    dst_mask = _args.out_mask + '/' + os.path.splitext(sfn)[0] + ".png"
                    shutil.copyfile(src_mask, dst_mask)

                # 复制viz图到存储目录
                if _args.out_viz:
                    if not os.path.exists(_args.out_viz):           # 文件夹是否存在
                        os.makedirs(_args.out_viz)

                    src_viz = "temp\label_viz.png"
                    dst_viz = _args.out_viz + '/' + os.path.splitext(sfn)[0] + ".png"
                    shutil.copyfile(src_viz, dst_viz)
            except:
                print(sfn)