import time
import cv2

class MyClass:
    ori_value = 2
    @classmethod
    def change_value(cls, w_value):
        cls.ori_value *= w_value
        print(cls.ori_value)

def costTime(func):
    print("costTime12")
    def dd(*args, **kwargs):
        print("dd14")
        start = time.time()
        C,H,W = func(*args, **kwargs)
        end = time.time()
        print("func costTime is:", end - start)
        print(C,H,W)
    return dd

#@costTime
def readImg(img_path):
    print("readImg24")
    img = cv2.imread(img_path)
    print("img.shape:", img.shape)
    return img.shape


if __name__ == '__main__':
    #img_path = r'C:\Users\1\Desktop\tmp\tmp_20230608\crop_no_crop_result\vpss_grp0_chn1_1920x1080_P420_1_1-1_crop.png'
    #readImg(img_path)
    #MyClass.my_static_method()
    print(MyClass.change_value(3))