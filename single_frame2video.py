import cv2
import os
import shutil
import json
#云端云雾分类数据
#从json文件中获取视频的烟的位置
#从视频/图片中截取烟并生成视频
def save_new_video(vid_path, point_list, save_path):
    for i, box in enumerate(point_list):
        left_up_point, right_down_point = (int(1920 * (box[1] - 0.5 * box[3])), int(1080 * (box[2] - 0.5 * box[4]))), (
        int(1920 * (box[1] + 0.5 * box[3])), int(1080 * (box[2] + 0.5 * box[4])))
        size = (right_down_point[0] - left_up_point[0], right_down_point[1] - left_up_point[1])

        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        avi_name = os.path.join(save_path, vid_path.split('\\')[-1].split('.')[0] + '_' + str(i) + '.mp4')
        avi_write = cv2.VideoWriter(avi_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

        while cap.isOpened():
            ret, img = cap.read()
            if ret:
                imgs = img[left_up_point[1]:right_down_point[1], left_up_point[0]:right_down_point[0], :]
                avi_write.write(imgs)
            else:
                break

        avi_write.release()
        cap.release()
        i += 1

def get_classification_video_from_xml(video_path, txt_label_path, save_result_path):
    video_list = [i for i in os.listdir(video_path) if i.endswith('mp4')]
    for video_file in video_list:
        txt_file = os.path.join(txt_label_path, video_file.replace('mp4', 'txt'))
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                files = f.readlines()
            point_list = []
            for lines in files:
                boxs= [float(i) for i in lines.split()]
                if boxs[0] == 0:
                    point_list.append(boxs)

            save_new_video(os.path.join(video_path, video_file), point_list, save_result_path)
        else:
            with open(os.path.join(video_path, 'data.txt'), 'a') as f:
                f.write(txt_file + '\n')

def copy_file_from_json_list(json_path, origal_path, obj_path):#获取本地视频文件列表后，获取云端json文件
    with open(json_path, 'r', encoding="utf-8", errors='ignore') as f:
        json_lists = f.readlines()
    for json_file in json_lists:
        json_file = json_file.strip('\n')
        ori_json_path = os.path.join(origal_path, json_file)
        obj_json_path = os.path.join(obj_path, json_file)

        if os.path.exists(ori_json_path):
            shutil.copy(ori_json_path, obj_json_path)

def get_classification_video_from_json(video_path, json_label_path, save_result_path):#centerxy
    video_list = [i for i in os.listdir(video_path) if i.endswith('mp4')]
    for k, video_file in enumerate(video_list):
        json_file = os.path.join(json_label_path, video_file.replace('mp4', 'json'))
        if os.path.exists(json_file):
            cap = cv2.VideoCapture(os.path.join(video_path, video_file))
            width = cap.get(3)  # float
            height = cap.get(4)  # float
            cap.release()
            with open(json_file, 'r', encoding='utf8') as fp:
                point_list = []
                json_data = json.load(fp)
                box_list = eval(json_data['extfireinfo'])
                for boxes in box_list :
                    h, w, x, y = boxes['codeh'], boxes['codew'], boxes['codex'], boxes['codey']
                    ratio_width = width / 1920
                    ratio_height = height / 1080
                    x = int(x * ratio_width)
                    y = int(y * ratio_height)
                    w = min(int(w * ratio_width), int(width - x - 1))
                    h = min(int(h * ratio_height), int(height - y - 1))
                    if 1 < w or 1 < h:
                        point_list.append([x, y, w, h])

                for i, box in enumerate(point_list):
                    #left_up_point, right_down_point = (box[0], box[1]), (box[0]+box[2], box[1]+box[3])
                    left_up_point, right_down_point = (int(box[0] - 0.5 * box[2]), int(box[1] - 0.5 * box[3])), (int(box[0] + 0.5 * box[2]), int(box[1] + 0.5 * box[3]))

                    size = (box[2], box[3])
                    cap = cv2.VideoCapture(os.path.join(video_path, video_file))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    avi_name = os.path.join(save_result_path, video_file.split('.')[0] + '_' + str(i) + '.mp4')
                    print(str(k), ":", avi_name)
                    avi_write = cv2.VideoWriter(avi_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                    while cap.isOpened():
                        ret, img = cap.read()
                        if ret:
                            imgs = img[left_up_point[1]:right_down_point[1], left_up_point[0]:right_down_point[0], :]
                            avi_write.write(imgs)
                        else:
                            break

                    avi_write.release()
                    cap.release()
                    i += 1

def get_classification_video_from_json_(video_path, json_label_path, save_result_path):#leftupxy
    video_list = [i for i in os.listdir(video_path) if i.endswith('mp4')]
    for k, video_file in enumerate(video_list):
        json_file = os.path.join(json_label_path, video_file.replace('mp4', 'json'))
        if os.path.exists(json_file):
            cap = cv2.VideoCapture(os.path.join(video_path, video_file))
            width = cap.get(3)  # float
            height = cap.get(4)  # float
            cap.release()
            with open(json_file, 'r', encoding='utf8') as fp:
                point_list = []
                json_data = json.load(fp)
                box_list = eval(json_data['extfireinfo'])
                for boxes in box_list :
                    h, w, x, y = boxes['codeh'], boxes['codew'], boxes['codex'], boxes['codey']
                    ratio_width = width / 1920
                    ratio_height = height / 1080
                    x = int(x * ratio_width)
                    y = int(y * ratio_height)
                    w = min(int(w * ratio_width), int(width - x - 1))
                    h = min(int(h * ratio_height), int(height - y - 1))
                    if 1 < w or 1 < h:
                        point_list.append([x, y, w, h])

                for i, box in enumerate(point_list):
                    left_up_point, right_down_point = (box[0], box[1]), (box[0]+box[2], box[1]+box[3])
                    #left_up_point, right_down_point = (int(box[0] - 0.5 * box[2]), int(box[1] - 0.5 * box[3])), (int(box[0] + 0.5 * box[2]), int(box[1] + 0.5 * box[3]))

                    size = (box[2], box[3])
                    cap = cv2.VideoCapture(os.path.join(video_path, video_file))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    avi_name = os.path.join(save_result_path, video_file.split('.')[0] + '_' + str(i) + '.mp4')
                    print(str(k), ":", avi_name)
                    avi_write = cv2.VideoWriter(avi_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
                    while cap.isOpened():
                        ret, img = cap.read()
                        if ret:
                            imgs = img[left_up_point[1]:right_down_point[1], left_up_point[0]:right_down_point[0], :]
                            avi_write.write(imgs)
                        else:
                            break

                    avi_write.release()
                    cap.release()
                    i += 1

def get_classification_video_from_json_1(video_path, json_label_path, save_result_path):
    video_list = [i for i in os.listdir(video_path) if i.endswith('mp4')]
    #video_list = video_list[5980:]
    for video_file in video_list:
        json_file = os.path.join(json_label_path, video_file.replace('mp4', 'json'))
        if os.path.exists(json_file):
            cap = cv2.VideoCapture(os.path.join(video_path, video_file))
            width = cap.get(3)  # float
            height = cap.get(4)  # float
            cap.release()
            with open(json_file, 'r', encoding='utf8') as fp:
                point_list = []
                json_data = json.load(fp)
                box_list = eval(json_data['extfireinfo'])
                for boxes in box_list :
                    h, w, x, y = boxes['codeh'], boxes['codew'], boxes['codex'], boxes['codey']
                    print(h, w, x, y)
                    ratio_width = width / 1920
                    ratio_height = height / 1080
                    x = int(x * ratio_width)
                    y = int(y * ratio_height)
                    print("x:", x)
                    print("y:", y)
                    w = min(int(w * ratio_width), int(width - x - 1))
                    h = min(int(h * ratio_height), int(height - y - 1))
                    print("w:", int(w * ratio_width), int(width - x - 1))
                    print("h:", int(h * ratio_width), int(height - y - 1))

                    if 1 < w or 1 < h:
                        point_list.append([x, y, w, h])

                for i, box in enumerate(point_list):
                    #left_up_point, right_down_point = (box[0], box[1]), (box[0] + box[2], box[1] + box[3])
                    left_up_point, right_down_point = (int(box[0] - 0.5 * box[2]), int(box[1] - 0.5 * box[3])), (int(box[0] + 0.5 * box[2]), int(box[1] + 0.5 * box[3]))

                    cap = cv2.VideoCapture(os.path.join(video_path, video_file))
                    k = 0
                    while cap.isOpened():
                        ret, img = cap.read()
                        k += 1
                        if ret:
                            cv2.rectangle(img, left_up_point, right_down_point,
                                          color=(0, 0, 255), thickness=2)
                            if k == 1:
                                cv2.imwrite(os.path.join(save_result_path, video_file.split('.')[0] + '_' + str(i)+ '.jpg'), img)

                            else:
                                break
                    cap.release()
                    i += 1

def batch_run(vid_list, json_label_path, saved_list):
    for i in range(len(vid_list)):
        get_classification_video_from_json_(vid_list[i], json_label_path, saved_list[i])

def cut_video(video_path, center_x, center_y, w, h, save_path):

    cap = cv2.VideoCapture(video_path)

    width = cap.get(3)  # float
    height = cap.get(4)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    cut_width = int(w * width)
    cut_height = int(h * height)

    x1 = int(center_x * width - 0.5 * cut_width)
    y1 = int(center_y * height- 0.5 * cut_height)

    x2  = int(center_x * width + 0.5 * cut_width)
    y2 = int(center_y * height + 0.5 * cut_height)

    cut_width = x2 - x1
    cut_height = y2 - y1

    print("===========")
    size = (cut_width, cut_height)
    #print("size:", size)
    print(save_path)

    avi_write = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    i = 0
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            i += 1
            #print(img.shape)
            #print(x1, y1, x2, y2)
            #print(y2 - y1, x2 - x1)
            imgs = img[y1:y2, x1:x2, :]
            #print(imgs.shape)
            avi_write.write(imgs)
            #break
        else:
            break
    print("all f:", i)
    avi_write.release()
    cap.release()


def batch_cut_video():
    video_path_list = ["cc20220819", "cc20220919", "cc20221018", "cc20221111", "cc20221212", "cc20230111", "cc20230222"]
    for i, video_p in enumerate(video_path_list):
        video_d = r'Z:\lisi\{}\烟'.format(video_p)
        label_d = r'Z:\qdm\single_frame_smoke_detection\train_dataset\2023_data\{}\labels'.format(video_p)
        save_d = r'F:\tmp_cut_video'
        for label_path in os.listdir(label_d):
            video_file = os.path.join(video_d, "{}.mp4".format(label_path.split('.')[0]))
            with open (os.path.join(label_d, label_path), 'r') as f:
                files = f.readlines()
            idx = 0
            for lines in files:
                _, left_x, up_y, w, h = [float(i) for i in lines.split()]
                save_file = os.path.join(save_d, "{}_{}.mp4".format(label_path.split('.')[0], idx))
                idx += 1
                cut_video(video_file, left_x, up_y, w, h, save_file)
            f.close()

if __name__ == '__main__':
    # video_path = r'Z:\lisi\cc20220121\正常'
    # json_label_path = r'E:\pycode\datasets\video_classification_dataset\normal_json20220902'
    # save_result_path = r'Z:\qdm\smoke_fog_classfication_video\cc20220121_normal_centerxy'
    # get_classification_video_from_json(video_path, json_label_path, save_result_path)
    #
    # vid_list = [r'Z:\lisi\cc20220121\正常', r'Z:\lisi\cc20220714\正常', r'Z:\lisi\cc20220819\正常']
    # saved_list = [r'Z:\qdm\smoke_fog_classfication_video\cc20220121_normal_leftupxy', r'Z:\qdm\smoke_fog_classfication_video\cc20220714_normal_leftupxy', r'Z:\qdm\smoke_fog_classfication_video\cc20220819_normal_leftupxy']
    # batch_run(vid_list, json_label_path, saved_list)
    batch_cut_video()