import time
import os
import glob
import datetime
import random
import sys

def demo():
    '''
    os.utime could only change file_modifed_time、file_access_time
    '''
    mtime = os.path.getmtime(__file__) #modifed_time
    ctime = os.path.getctime(__file__) #make_time
    print(time.gmtime(mtime))
    print(time.gmtime(ctime))

    start_time = datetime.datetime(1970, 1, 1, 8, 0, 0)#1970_1_1_8:00:00
    end_time = datetime.datetime(2023, random.randint(1, 12), random.randint(1, 28), random.randint(0, 23), random.randint(0, 60), random.randint(0, 60))#2023_anyTime
    time_delta = (end_time - start_time).total_seconds()#export_secs
    #print(time_delta)

    dst_access_time = time.time()
    dst_mofifed_time = time_delta

    #os.utime(file_path, (dst_access_time, dst_mofifed_time))#access_time, modifed_time

def batch_modify_modifiedTime():
    file_dir = r'C:\Users\1\Desktop\tmp\metric\change_file_time\小试平台巡检单'
    file_list = glob.glob(os.path.join(file_dir + '\*.xls'))
    for file in file_list:
        mtime = os.path.getmtime(file) #modifed_time
        ctime = os.path.getctime(file) #make_time

        if file != r"C:\Users\1\Desktop\tmp\metric\change_file_time\小试平台巡检单\东北虎豹系统巡检单20230407-202304113.xls":
            dst_time = os.path.basename(file).split('.')[0][-8:]
        else:
            dst_time = "20230413"

        print(dst_time)

        start_time = datetime.datetime(1970, 1, 1, 8, 0, 0)

        end_time = datetime.datetime(int(dst_time[:4]), int(dst_time[4:6]), int(dst_time[6:8]), random.randint(14, 16), random.randint(0, 60), random.randint(0, 60))

        time_delta = end_time - start_time

        print(time_delta.total_seconds())

        os.utime(file, (time.time(), time_delta.total_seconds()))#access_time, modifed_time


if __name__ == '__main__':
    demo()