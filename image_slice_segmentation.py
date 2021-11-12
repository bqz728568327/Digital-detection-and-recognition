# -*- coding: utf-8 -*-
# @Time    : 2021-11-11
# @Author  : Bai
# @Email   : 728568327@qq.com
# @File    : image_slice_segmentation.py

import os
import cv2
import numpy as np
from config_parameter import Config
from config_logging import LogConfig

def cutting_segmentation_image(image):
    fragment_image_list = []
    disorder_slice_image_list = []
    kernel = np.ones((2, 2), np.uint8)
    open_operation_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    gray_image = cv2.cvtColor(open_operation_image, cv2.COLOR_BGR2GRAY)
    gaussian_blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    median_blur_image = cv2.medianBlur(gaussian_blur_image, 3)
    binary_image = cv2.adaptiveThreshold(median_blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
    result_binary_image = cv2.adaptiveThreshold(gaussian_blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
    results = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    status = results[2]
    for i in range(1, len(results[2])):
        x, y, w, h, s = status[i]
        if w <= 28 and w >= 5 and h <= 25 and h >= 10 and s >= 35:
            # 在兴趣区域显示
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            disorder_slice_image_list.append({'x':x, 'y':y, 'w':w, 'h':h})
    disorder_slice_image_list = sorted(disorder_slice_image_list, key=lambda x: x['x'])
    for i in range(4, len(disorder_slice_image_list)):
        slice = disorder_slice_image_list[i]
        x = slice['x']
        y = slice['y']
        w = slice['w']
        h = slice['h']
        fragment_image_list.append(result_binary_image[y:y + h, x:x + w])
    return fragment_image_list

def get_max_segmentation_image(image):
    kernel = np.ones((2,2), np.uint8)
    dilation_image = cv2.dilate(image, kernel, 1)
    erosion_image = cv2.erode(dilation_image, kernel, 3)
    gray_image = cv2.cvtColor(erosion_image, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray_image, (5, 5), 0)
    low_h, high_h, min_x, max_x = get_interest_image(blur_gray)
    return image[low_h:high_h, min_x:max_x]

def get_interest_image(image):
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    # 确定下界
    high_h = get_low_boundary(binary_image)
    # 8连通搜索
    results = cv2.connectedComponentsWithStats(binary_image[0:high_h, :], connectivity=8)
    status = results[2]
    index_y_count = {}
    threshold = 5
    for i in range(1, len(results[2])):
        x, y, w, h, s = status[i]
        flag = 1
        if w <= 28 and w >= 5 and h <= 25 and h >= 10 and s >= 40:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            for key in index_y_count.keys():
                if abs(y - key) < threshold:
                    index_y_count[key] = index_y_count[key] + 1
                    flag = 0
                    break
            if flag:
                index_y_count[y] = 1
    # 获取最大可能性上界
    low_h = 0
    count = 0
    for key in index_y_count.keys():
        if index_y_count[key] > count:
            count = index_y_count[key]
            low_h = key
    # 上界阈值
    if low_h - threshold >= 0:
        low_h -= threshold
    results = cv2.connectedComponentsWithStats(binary_image[low_h:high_h, :], connectivity=8)
    status = results[2]
    temp_image = image[low_h:high_h, :]
    index_x_count = []
    for i in range(1, len(results[2])):
        x, y, w, h, s = status[i]
        if w <= 28 and w >= 5 and h <= 25 and h >= 10 and s >= 40:
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            index_x_count.append(x)
            index_x_count.append(x+w)
    min_x = min(index_x_count)
    max_x = max(index_x_count)
    # cv2.imshow('binary', binary_image)
    # cv2.imshow('temp', temp_image)
    return low_h, high_h, min_x, max_x

def get_low_boundary(binary_image):
    h_h = get_h_line(binary_image)
    h_low = 0
    h_mid = int(len(h_h) / 3)
    average = h_h[h_mid + 1]
    for i in range(h_mid):
        if h_h[h_mid - i] > average + 50:
            h_low = h_mid - i
            break
    return h_low

def get_h_line(binary_image):
    h, w = binary_image.shape[:2]
    start_w = int(w / 4)
    end_w = w - start_w
    h_h = [0] * h
    # 水平投影观察
    hprojection = np.zeros(binary_image.shape, dtype=np.uint8)
    hprojection.fill(255)
    for j in range(h):
        for i in range(start_w, end_w):
            if binary_image[j, i] == 0:
                h_h[j] += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j, i] = 0
    cv2.imshow('hpro', hprojection)
    return h_h

def get_fragment_label(filename) -> list:
    fragment_label_list = []
    n = len(filename)
    for i in range(n):
        c = filename[i]
        if (ord(c) >= ord('0') and ord(c) <= ord('9') ):
            fragment_label_list.append(filename[i])
        if ord(c) == ord('X'):
            fragment_label_list.append('10')
    return fragment_label_list

def save_train_data(output_root, number, fragment_image_list, fragment_label_list):
    output_dir = os.path.join(output_root, str(number))
    if len(fragment_image_list) != len(fragment_label_list):
        raise Exception()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(len(fragment_image_list)):
        output = os.path.join(output_dir, str(i) + '-' + fragment_label_list[i] + '.png')
        fragment_image = fragment_image_list[i]
        resize_fragment_image = cv2.resize(fragment_image, (28,28))
        cv2.imwrite(output, resize_fragment_image)

cfg = Config()
logging = LogConfig()

if __name__ == '__main__':

    is_train = cfg.is_train

    if is_train:
        root_dir = cfg.train_rotated_image_root
        output_dir = cfg.train_slice_image_root
    else:
        root_dir = cfg.test_rotated_image_root
        output_dir = cfg.test_slice_image_root


    list = os.listdir(root_dir)

    logging.debug(f'''Print Segmentation Parameters:
            Train and Test Model:      {cfg.is_train}
            Image input root:          {root_dir}
            Image output root:         {output_dir}
            Image size:                {len(list)}
        ''')

    error_nums = 0
    for i in range(len(list)):
        filename = list[i]
        if i % len(list) == i:
            file_path = os.path.join(root_dir, filename)
            # 读入图片
            image = cv2.imread(file_path)
            # 对于反色图片处理
            # if i % len(list) == 6:
                # image = cv2.bitwise_not(image)
            # 统一图片大小
            resize_image = cv2.resize(image, (512, 256))
            # 获取兴趣区域
            interest_image = get_max_segmentation_image(resize_image)
            # 数字图像切割
            fragment_image_list = cutting_segmentation_image(interest_image)
            # 图像标签切割
            fragment_label_list = get_fragment_label(filename)
            # 保存中间过程
            try:
                save_train_data(output_dir, i, fragment_image_list, fragment_label_list)
                logging.info('No.{:<3} Image Cutting Success : {},  '.format(i + 1, filename))
            except:
                logging.error('No.{:<3} Image Cutting Process have a Error : {},  '.format(i + 1, filename))
                error_nums += 1
            # cv2.imshow('a'+str(i), interest_image)
    logging.debug('Cutting Image total : {} ,Among them, the errors nums is : {}'.format(len(list), error_nums))
    logging.debug('All Images have been Cut and the Output Path is : {}'.format(output_dir))


    cv2.waitKey()
    cv2.destroyAllWindows()