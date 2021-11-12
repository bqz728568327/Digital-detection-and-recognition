# -*- coding: utf-8 -*-
# @Time    : 2021-11-11
# @Author  : Bai
# @Email   : 728568327@qq.com
# @File    : image_rotating.py

import cv2
import numpy as np
import os
from config_parameter import Config
from config_logging import LogConfig

def rotate(image, threshold, minLineLength, maxLineGap, flag=False):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    blur_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    edges = cv2.Canny(blur_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    height, width = image.shape[:2]
    # 寻找目标线
    target_x1, target_y1, target_x2, target_y2 = 0,0,0,0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < abs(x1 - x2) and y2 < height // 3 and y1 < height // 3 and abs(x2 - x1) >= width // 3:
            if y1 > target_y1:
                target_x1, target_y1, target_x2, target_y2 = x1, y1, x2, y2
    # cv2.line(image, (target_x1, target_y1), (target_x2, target_y2), (0, 255, 0), 2)
    #       获取旋转角度
    angle = cv2.fastAtan2(float((target_y2 - target_y1)), float((target_x2 - target_x1)))
    angle = angle % 90
    if angle > 45:
        angle = angle - 90
    rotate_mat = cv2.getRotationMatrix2D((width/2,height/2), angle, 1.0)  # 获取旋转矩阵
    background_color = abstract_background_color(gray_image)
    rotate_image = cv2.warpAffine(image, rotate_mat, (width, height), borderValue=background_color)
    # cv2.line(rotate_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return rotate_image

def abstract_background_color(image, thresh=0):
    h, w = image.shape[:2]
    num_size = h*w
    b = g = r = 0
    for i in range(h):
        for j in range(w):
            b += image[i, j, 0]
            g += image[i, j, 1]
            r += image[i, j, 2]
    return (int(b // num_size + thresh), int(g // num_size + thresh), int(r // num_size + thresh))

cfg = Config()
logging = LogConfig()

if __name__ == '__main__':

    is_train = cfg.is_train

    if cfg.is_train:
        image_root = cfg.train_image_root
        output_root = cfg.train_rotated_image_root
    else:
        image_root = cfg.test_image_root
        output_root = cfg.test_rotated_image_root

    if not os.path.exists(output_root):
        os.mkdir(output_root)
    list = os.listdir(image_root)

    logging.debug(f'''Print Rotating Parameters:
        Train and Test Model:      {cfg.is_train}
        Image input root:          {image_root}
        Image output root:         {output_root}
        Image size:                {len(list)}
    ''')

    for i in range(len(list)):
        filename = list[i]
        if i % 100 == i:
            image_path = os.path.join(image_root, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (512, 256))
            rotated_image = rotate(image, 50, 50, 80)
            # cv2.imshow("a"+str(i), rotated_image)
            output = os.path.join(output_root, filename)
            cv2.imwrite(output, rotated_image)
            logging.info('No.{:<3} Image Rotated Success : {}'.format(i+1, filename))
    logging.debug('All Images have been Rotated and the Output Path is : {}'.format(output_root))

