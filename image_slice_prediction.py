# -*- coding: utf-8 -*-
# @Time    : 2021-11-11
# @Author  : Bai
# @Email   : 728568327@qq.com
# @File    : image_slice_prediction.py

import os
import logging
import torch
from PIL import Image
import numpy as np
from model.rnn import RNN
from model.cnn import Lenet
from config_parameter import Config
from config_logging import LogConfig


def rnn_base_learner(slice_image, rnn_net):
    np_array_img = np.array(slice_image)
    np_array_img = np.expand_dims(np_array_img, axis=2).transpose((2, 0, 1)).astype(float)
    input = torch.FloatTensor(np_array_img)
    output = rnn_net(input)
    pre_lab = torch.argmax(output, 1)
    return pre_lab.int().item()

def cnn_base_learner(slice_image, cnn_net):
    np_array_img = np.array(slice_image)
    np_array_img = np.expand_dims(np_array_img, axis=2).transpose((2, 0, 1)).astype(float)
    np_array_img = np.expand_dims(np_array_img, axis=3).transpose((3, 0, 1, 2)).astype(float)
    input = torch.FloatTensor(np_array_img)
    output = cnn_net(input)
    pre_lab = torch.argmax(output, 1)
    return pre_lab.int().item()

def knn_base_learner(slice_image, data_set, label_set_list, k=10):
    n = len(image_set_list)
    image_array = np.array(slice_image, dtype=np.uint8).ravel()
    diffMat = np.tile(image_array, (n, 1)) - data_set
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances ** 0.5
    sortedDistIndicies = distance.argsort()
    #     classCount保存的K是魅力类型   V:在K个近邻中某一个类型的次数
    classCount = {}
    for i in range(k):
        # 获取对应的下标的类别
        voteLabel = label_set_list[sortedDistIndicies[i]]
        # 给相同的类别次数计数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]

def integrator_prediction_vote(slice_image, rnn_net, cnn_net, data_set, label_set_list, k=3):
    pre_vote = {}
    # cnn-net 预测()
    cnn_pre_lab = int(cnn_base_learner(slice_image, cnn_net))
    # rnn—net 预测
    rnn_pre_lab = int(rnn_base_learner(slice_image, rnn_net))
    # 模板匹配预测
    knn_pre_lab = int(knn_base_learner(slice_image, data_set, label_set_list, k))
    if cfg.cnn_learner_open:
        if cnn_pre_lab not in pre_vote.keys():
            pre_vote[cnn_pre_lab] = 1
        else:
            pre_vote[cnn_pre_lab] = pre_vote[cnn_pre_lab] + 1
    if cfg.rnn_learner_open:
        if rnn_pre_lab not in pre_vote.keys():
            pre_vote[rnn_pre_lab] = 1
        else:
            pre_vote[rnn_pre_lab] = pre_vote[rnn_pre_lab]+1
    if cfg.knn_learner_open:
        if knn_pre_lab not in pre_vote.keys():
            pre_vote[knn_pre_lab] = 1
        else:
            pre_vote[knn_pre_lab] = pre_vote[knn_pre_lab] + 1

    keys = list(pre_vote.keys())
    values = list(pre_vote.values())
    # print(keys, '--------', values)
    top_tic_index = values.index(max(values))
    return keys[top_tic_index]

def get_feature_vector(image_slice_dir):
    if not os.path.exists(image_slice_dir):
        raise Exception ("Not find target dir {}".format(image_slice_dir))
    image_set_list = []
    label_set_list = []
    list = os.listdir(image_slice_dir)
    for cur_dir in list:
        cur_path = os.path.join(image_slice_dir, cur_dir)
        cur_list = os.listdir(cur_path)
        for filename in cur_list:
            cur_label = filename.split(".")[0].split('-')[-1]
            cur_image_path = os.path.join(cur_path, filename)
            cur_image = Image.open(cur_image_path).convert('L')
            np_array_image = np.array(cur_image, dtype=np.uint8).ravel()
            image_set_list.append(np_array_image)
            label_set_list.append(cur_label)
    return image_set_list, label_set_list

def bubble_sort_flag(sort_list, label_list, pre_list):
    n = len(sort_list)
    for index in range(n):
        flag = True
        for j in range(1, n - index):
            if int(sort_list[j-1]) > int(sort_list[j]):
                sort_list[j - 1], sort_list[j] = sort_list[j], sort_list[j - 1]
                label_list[j - 1], label_list[j] = label_list[j], label_list[j - 1]
                pre_list[j - 1], pre_list[j] = pre_list[j], pre_list[j - 1]
                flag = False
        if flag:
            # 没有发生交换，直接返回list
            return label_list, pre_list
    return label_list, pre_list

cfg = Config()
logging = LogConfig()

if __name__ == '__main__':

    root_dir = cfg.test_slice_image_root

    # rnn-net 加载模型
    rnn_net = RNN(cfg.rnn_in_channels, cfg.rnn_hidden_channels, cfg.rnn_layer_channels, cfg.rnn_n_classes)
    rnn_net.load_state_dict(torch.load(cfg.rnn_model_save_path))

    # cnn-net 加载模型
    cnn_net = Lenet(cfg.cnn_in_channels, cfg.cnn_n_classes)
    cnn_net.load_state_dict(torch.load(cfg.cnn_model_save_path))

    # knn 准备数据集
    image_set_list, label_set_list = get_feature_vector(cfg.train_slice_image_root)
    data_set = np.array(image_set_list)

    logging.debug(f'''Integrator System:
    CNN Base Learner: {cfg.cnn_learner_open}
    RNN Base Learner: {cfg.rnn_learner_open}
    KNN Base Learner: {cfg.knn_learner_open}
    ''')

    dir_list = os.listdir(root_dir)
    total = 0
    corrects = 0
    recall = 0
    image_num = len(dir_list)
    for i in range(len(dir_list)):
        target_path = os.path.join(root_dir, dir_list[i])
        slice_list = os.listdir(target_path)
        pre_list = []
        label_list = []
        sort_list = []
        for filename in slice_list:
            filepath = os.path.join(target_path, filename)
            slice_image = Image.open(filepath).convert('L')
            pre_lab = integrator_prediction_vote(slice_image, rnn_net, cnn_net, data_set, label_set_list, k=1)
            pre_list.append(pre_lab)
            label_list.append(filename.split('.')[0].split('-')[-1])
            sort_list.append(filename.split('.')[0].split('-')[0])
        label_list, pre_list = bubble_sort_flag(sort_list, label_list, pre_list)
        pre_str = ''
        label_str = ''
        for k in range(len(pre_list)):
            if str(pre_list[k]) == str(10):
                pre_str += 'X'
                label_str += 'X'
            else:
                pre_str += str(pre_list[k])
                label_str += str(label_list[k])
            # 计算acc
            if str(pre_list[k]) == str(label_list[k]):
                corrects += 1
            # 总数量
            total += 1
        if pre_str == label_str:
            recall += 1
        logging.info('第 {:<3}张图片，标签： {:<13} ---- 预测：{:<13} ---- 一致性：{}'.format(i+1,label_str, pre_str, label_str == pre_str))

    image_list = os.listdir(cfg.test_image_root)
    label_nums = 0
    image_total_num = len(image_list)
    for filename in image_list:
        n = len(filename)
        for i in range(n):
            c = filename[i]
            if (ord(c) >= ord('0') and ord(c) <= ord('9')) or ord(c) == ord('X'):
                label_nums += 1

    logging.debug(f'''Statistics dashboard:
    Image nums / Image total nums:       {len(dir_list)} / {image_total_num}
    Image slices / Image total labels :  {corrects} / {label_nums} 
    Image Precision rate :               {corrects / label_nums * 100}
    Image Recall rate :                  {recall / image_total_num * 100}
    ''')