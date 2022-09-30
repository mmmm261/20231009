#Usage
#获取视频列表中所有视频第一帧并保存
import os
import cv2
import glob
import random

def get_random_videos(file_path):
    random.seed(2022)
    file_list = os.listdir(file_path)
    random_list = random.sample(range(1, len(file_list)), 1000)

    file_lists = []
    for i in random_list:
        file_lists.append(os.path.join(file_path, file_list[i]))
    return file_lists

def get_first_frame(mp4_path, save_path):
    video_list = glob.glob(mp4_path + '\\*.mp4')
    video_list = video_list[100:1000]

    for file in video_list:
        videoCapture = cv2.VideoCapture(file)

        success, frame = videoCapture.read()

        new_path = os.path.join(save_path, "{}.jpg".format(file.split('\\')[-1].split('.')[0]))
        print(new_path)
        cv2.imwrite(new_path, frame)

def get_specific_frame(mp4_path, save_path):
    video_list = glob.glob(mp4_path + '\\*.mp4')
    idx = 0

    for file in video_list:
        videoCapture = cv2.VideoCapture(file)
        new_path = os.path.join(save_path, "{}.jpg".format(file.split('\\')[-1].split('.')[0]))
        specific_idx = videoCapture.get(7)
        specific_idx = int(specific_idx / 2)
        while True:
            success, frame = videoCapture.read()
            idx = idx + 1
            if idx == specific_idx:
                cv2.imwrite(new_path, frame)
                break
            if not success:
                break

def cut_video(video_path, save_path, start_nums):
    video = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')#XVID, 'MPEG'

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps > 25 or fps < 1:
        fps = 25;

    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    savepath = os.path.join(save_path, "{}.avi".format(0))

    out = cv2.VideoWriter(savepath, fourcc, fps, size)

    frameToStart = start_nums

    video.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)

    count = 0
    while video.isOpened():
        ret, frame = video.read()  # 捕获一帧图像

        if ret:
            print(count)
            if count < 20 * 25:
                out.write(frame)
                count += 1
            else:
                break
        else:
            break
        if cv2.waitKey(1) == 27 & 0xFF:
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = r'E:\Program Files\feiq\Recv Files\20220710_test_video\test_1\test_1.mp4'
    save_path = r'E:\Program Files\feiq\Recv Files\20220710_test_video\test_1'

    cut_video(video_path, save_path, 85 * 25)