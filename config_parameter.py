# -*- coding: utf-8 -*-
# @Time    : 2021-11-11
# @Author  : Bai
# @Email   : 728568327@qq.com
# @File    : config_parameter.py

class Config:

    def __init__(self):
        super(Config, self).__init__()

        self.is_train = False
        self.train_image_root = './data/train_data/train_images'
        self.train_rotated_image_root = './data/train_data/rotated_images'
        self.train_slice_image_root = './data/train_data/slice_images'
        self.test_image_root = './data/test_data/test_images'
        self.test_rotated_image_root = './data/test_data/rotated_images'
        self.test_slice_image_root = './data/test_data/slice_images'

        # 预测集成器 参数配置
        self.cnn_learner_open = True
        self.rnn_learner_open = True
        self.knn_learner_open = True

        # LeNet-5 网络配置参数
        self.cnn_model = 'LeNet-5'
        self.cnn_epochs = 100
        self.cnn_batch_size = 64
        self.cnn_val_percent = 0.1      # range(0,1)
        self.cnn_in_channels = 1
        self.cnn_n_classes = 11
        self.cnn_optimizer = 'Adam'
        self.cnn_lr = 0.0003
        self.cnn_weight_decay = 0
        self.cnn_criterion = 'CrossEntropyLoss'
        self.cnn_model_save_path = 'points/cnn_model_parameter.pth'

        # RNN 网络配置参数
        self.rnn_model = 'RNN'
        self.rnn_model = 'LeNet-5'
        self.rnn_epochs = 100
        self.rnn_batch_size = 64
        self.rnn_val_percent = 0.1  # range(0,1)
        self.rnn_in_channels = 28
        self.rnn_hidden_channels = 128
        self.rnn_layer_channels = 1
        self.rnn_n_classes = 11
        self.rnn_optimizer = 'Adam'
        self.rnn_lr = 0.0005
        self.rnn_weight_decay = 0
        self.rnn_criterion = 'CrossEntropyLoss'
        self.rnn_model_save_path = 'points/rnn_model_parameter.pth'

