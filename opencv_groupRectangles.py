import numpy as np
import cv2

points = np.array([[ 76, 620, 107, 635],
       [ 87, 540,  96, 554],
       [ 77, 459, 100, 473],
       [ 77, 377, 107, 392],
       [ 77, 297, 100, 311],
       [ 68, 217, 101, 231],
       [ 86, 139,  94, 147],
       [ 68, 135, 107, 150],
       [ 69,  54, 107,  69],
       [ 77,  54,  97,  69],
       [545, 641, 580, 655],
       [454, 640, 489, 655],
       [364, 641, 399, 655],
       [375, 641, 399, 655],
       [364, 640, 373, 654],
       [636, 638, 671, 655],
       [647, 641, 671, 655],
       [660, 643, 667, 651],
       [273, 638, 309, 655],
       [273, 641, 284, 655],
       [284, 640, 309, 655],
       [183, 638, 214, 655],
       [183, 641, 196, 655],
       [105, 637, 115, 655],
       [ 77, 625, 102, 634]])

points= np.array([[50,50,100,100],
                  [128,128,384,384],
                  [256,256,480,480],
                  [150,400,200,450]])

img = np.zeros([1024,1024,3])
for pred_point in points:

       cv2.rectangle(img, (pred_point[0], pred_point[1]), (pred_point[2], pred_point[3]), color=(0, 0, 255),
                     thickness=1)

cv2.imwrite(r'C:\Users\1\Desktop\label_cla\groupRectangles.png', img)

def groupRect(rectarray, rectthreshold=1, eps=0.1):
    """
    Groups similar rectangles together in rectangle array \n
    Input : Rectangle Array \n
    Output : Rectangle Array without split parts
    """
    results = cv2.groupRectangles(np.concatenate((rectarray, rectarray)),rectthreshold,eps=eps)[0]
    results = [[group] for group in results]
    return np.array(results)

results = cv2.groupRectangles(np.concatenate((points, points)),2,eps=0.05)[0]
img2 = np.zeros([1024,1024,3])
for r in results:
    cv2.rectangle(img2,(r[0], r[1]),(r[2], r[3]),(255,0,0),1)

cv2.imshow('result', img2)

cv2.waitKey(0)
#cv2.imwrite(r'C:\Users\1\Desktop\label_cla\groupRectangles_result.png', img2)
