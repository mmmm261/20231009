import os
import cv2

def draw_box_from_json(json_path, img_path, result_save_path):
    with open(json_path, 'r', encoding="utf-8", errors='ignore') as f:
        json_lists = f.readlines()
    json_lists = eval(json_lists[0])
    imgs = []
    for json_file in json_lists:
        box = json_file['bbox']
        img_path = os.path.join(img_path, "{}.jpg".format(str(json_file['image_id']).zfill(12)))
        imgs.append(img_path)

        if len(imgs) == 1:
            img = cv2.imread(img_path)
        else:
            if img_path != imgs[-2]:
                img = cv2.imread(img_path)
        result = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color=(0, 0, 255), thickness=2)
        result_path = os.path.join(result_save_path, "{}.jpg".format(str(json_file['image_id']).zfill(12)))
        cv2.imwrite(result_path, result)

def draw_segmentation_points(json_path, img_path, result_save_path):
    with open(json_path, 'r', encoding="utf-8", errors='ignore') as f:
        json_lists = f.readlines()
    json_lists = eval(json_lists[0])
    imgs = []
    points_list = []
    idx = json_lists[0]['image_id']
    for json_file in json_lists:

        points = [json_file['segmentation']]
        img_path_ = os.path.join(img_path, "{}.jpg".format(str(json_file['image_id']).zfill(12)))
        imgs.append(img_path_)

        if json_file['image_id'] == idx:
            points_list.append(points[0])
            print(points[0])

        else:
            print("---------------------")
            img = cv2.imread(imgs[-2]) #if len(imgs>1) else cv2.imread(imgs[0])
            for points_ in points_list:

                point = points_[0]
                print(point)
                for i in range(int(len(point) / 2)):
                    # print((points[2*i], points[2*i+1]))
                    cv2.circle(img, ((int(point[2 * i]), int(point[2 * i + 1]))), radius=1, color=(0, 0, 255),
                               thickness=2)
            result_path = os.path.join(result_save_path, imgs[-2].split('\\')[-1])

            cv2.imwrite(result_path, img)
            idx = json_file['image_id']
            points_list = []
            points_list.append(points[0])

def draw_segmentation_lines(json_path, img_path, result_save_path):
    with open(json_path, 'r', encoding="utf-8", errors='ignore') as f:
        json_lists = f.readlines()
    json_lists = eval(json_lists[0])
    imgs = []
    points_list = []
    idx = json_lists[0]['image_id']
    for json_file in json_lists:

        points = [json_file['segmentation']]
        img_path_ = os.path.join(img_path, "{}.jpg".format(str(json_file['image_id']).zfill(12)))
        imgs.append(img_path_)

        if json_file['image_id'] == idx:
            points_list.append(points[0])
            print(points[0])

        else:
            print("---------------------")
            img = cv2.imread(imgs[-2]) #if len(imgs>1) else cv2.imread(imgs[0])
            for points_ in points_list:

                point = points_[0]
                print(point)
                for i in range(int(len(point) / 2)):
                    # print((points[2*i], points[2*i+1]))
                    if 2 * i + 3 <= len(point):
                        cv2.line(img, (int(point[2 * i]), int(point[2 * i + 1])), (int(point[2 * i + 2]), int(point[2 * i + 3])), (0, 255, 0), 2)

            result_path = os.path.join(result_save_path, imgs[-2].split('\\')[-1])

            cv2.imwrite(result_path, img)
            idx = json_file['image_id']
            points_list = []
            points_list.append(points[0])
if __name__ == '__main__':
    json_path = r'E:\pycode\PolyWorldPretrainedNetwork-main\predictions.json'
    img_path = r'E:\pycode\PolyWorldPretrainedNetwork-main\val\images'
    result_save_path = r'E:\pycode\PolyWorldPretrainedNetwork-main\val\seg_result'
    draw_segmentation_lines(json_path, img_path, result_save_path)