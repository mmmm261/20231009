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

def get_first_frame(video_path, save_path):

    videoCapture = cv2.VideoCapture(video_path)

    success, frame = videoCapture.read()

    if success:
        cv2.imwrite(save_path, frame)
    else:
        print("read video error:", video_path)

def get_specific_frame(video_path, save_path):

    videoCapture = cv2.VideoCapture(video_path)

    #specific_idx = int(videoCapture.get(7) / 2)
    #specific_idx = 10
    specific_idx = 25
    idx = 0
    while True:
        success, frame = videoCapture.read()
        if not success:
            break
        if idx == specific_idx:
            cv2.imwrite(save_path, frame)
            break
        idx = idx + 1

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

def merge_video(origal_vid_path, save_vid_path):
    cap = cv2.VideoCapture(origal_vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 25 or fps < 1:
        fps = 25

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #avi_name = os.path.join(save_path, vid_path.split('\\')[-1].split('.')[0] + '_' + str(i) + '.mp4')
    avi_write = cv2.VideoWriter(save_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)


    count = 0
    while cap.isOpened():
        ret, frame = cap.read()  # 捕获一帧图像

        if ret:
            print(count)
            if count < 10 * 25:
                avi_write.write(frame)
                count += 1
            else:
                break
        else:
            cap = cv2.VideoCapture(origal_vid_path)
        if cv2.waitKey(1) == 27 & 0xFF:
            break
    cap.release()
    avi_write.release()
    cv2.destroyAllWindows()

def batch_get_first_frame(video_dir, save_dir):

    video_list = [i for i in os.listdir(video_dir) if i.endswith('mp4')]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if len(video_list) > 0:
        print("current list len:{}".format(len(video_list)))
        for i, video_file in enumerate(video_list):
            save_path = os.path.join(save_dir, "{}.jpg".format(video_file.split('\\')[-1].split('.')[0]))
            get_first_frame(os.path.join(video_dir, video_file), save_path)
            if i % 500 == 0:
                print("process:{}:".format(i), os.path.join(video_dir, video_file))
    else:
        print("current list is empty".format(video_dir))

def get_single_frame_data_20230306():
    video_path_list = ["cc20220819", "cc20220919", "cc20221018", "cc20221111", "cc20221212", "cc20230111", "cc20230222"]
    for video_p in video_path_list:
        video_d = r'Z:\lisi\{}\烟'.format(video_p)
        save_d = r'F:\tmp_single_frame_data\{}'.format(video_p)
        print("current process dir:", video_d)
        print("current save dir:", save_d)
        batch_get_first_frame(video_d, save_d)

def write_single_frame_to_txt(img_path_list, txt_save_path):
    with open (txt_save_path, 'a') as f:
        for img_path in img_path_list:
            image_path = os.path.join(img_path, "images")
            label_path = os.path.join(img_path, "labels")
            file_list = [i.split('.')[0] for i in os.listdir(image_path) if i.endswith('jpg')]
            for file in file_list:
                image_file = os.path.join(image_path, "{}.jpg".format(file))
                label_file = os.path.join(label_path, "{}.txt".format(file))
                f.write(image_file + "\t" + label_file + "\t" + "smoke" + "\n")
    f.close()

def get_single_frame_data_20230321():
    video_path_list = ["cc20220819", "cc20220919", "cc20221018", "cc20221111", "cc20221212", "cc20230111", "cc20230222"]
    for i, video_p in enumerate(video_path_list):
        video_d = r'Z:\lisi\{}\正常'.format(video_p)
        print("current process dir:", video_d)
        save_d = r'F:\tmp_normal_smoke\{}'.format(video_p)
        print("current save dir:", save_d)

        if not os.path.exists(save_d):
            os.mkdir(save_d)
        if video_p is not "cc20230222":
            batch_get_first_frame(video_d, save_d)
        else:
            video_list_ = [i for i in os.listdir(video_d) if i.endswith('mp4')]
            video_list0 = [i for i in video_list_ if not i.startswith('ts-20201112')]
            video_list1 = [i for i in video_list_ if i.startswith('ts-20201112')][:1195]
            video_list = video_list1 + video_list0

            if not os.path.exists(save_d):
                os.mkdir(save_d)

            if len(video_list) > 0:
                print("current list len:{}".format(len(video_list)))
                for i, video_file in enumerate(video_list):
                    save_path = os.path.join(save_d, "{}.jpg".format(video_file.split('\\')[-1].split('.')[0]))
                    get_first_frame(os.path.join(video_d, video_file), save_path)
                    if i % 500 == 0:
                        print("process:{}:".format(i), os.path.join(video_d, video_file))
            else:
                print("current list is empty".format(video_d))

def get_single_frame_fog_data_20230817():
    #print("here")
    video_dir = r'Z:\lisi\cc20230222\fog'
    img_save_dir = r'F:\dataset\cc20230222_fog'
    for i, video_file in enumerate(os.listdir(video_dir)):
        print(i, ":", video_file)
        video_path = os.path.join(video_dir, video_file)
        img_path = os.path.join(img_save_dir, video_file.replace('mp4', 'jpg'))
        get_specific_frame(video_path, img_path)




if __name__ == '__main__':
    # img_dir = r'Z:\qdm\single_frame_smoke_detection\train_dataset'
    # img_list = [os.path.join(img_dir, i) for i in os.listdir(img_dir)]
    # txt_path = r'Z:\qdm\single_frame_smoke_detection\train_data.txt'
    # write_single_frame_to_txt(img_list, txt_path)
    # get_single_frame_fog_data_20230817()
    for file in os.listdir(r'C:\Users\1\Desktop\fog_20230907\video'):
        video_path = os.path.join(r'C:\Users\1\Desktop\fog_20230907\video', file)
        save_path = video_path.replace('video', 'Image').replace('mp4', 'jpg')
        get_specific_frame(video_path, save_path)




