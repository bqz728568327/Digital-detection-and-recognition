# -*- coding: utf-8 -*-
# @Time    : 2021-11-11
# @Author  : Bai
# @Email   : 728568327@qq.com
# @File    : train_model_rnn.py

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from config_parameter import Config
from config_logging import LogConfig
from model.rnn import RNN

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_dir = cfg.train_slice_image_root
    model_save_path = cfg.rnn_model_save_path

    logging.debug(f'''Network::
        model:           {cfg.rnn_model}
        input channels:  {cfg.rnn_in_channels}
        hidden channels: {cfg.rnn_hidden_channels}
        layer channels:  {cfg.rnn_layer_channels}
        output channels: {cfg.rnn_n_classes}
        ''')

    dataset = ImageDataset(images_dir)
    val_percent = cfg.rnn_val_percent
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train,
                              batch_size=cfg.rnn_batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(val,
                            batch_size=cfg.rnn_batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True)

    net = RNN(cfg.rnn_in_channels, cfg.rnn_hidden_channels, cfg.rnn_layer_channels, cfg.rnn_n_classes)

    if cfg.rnn_optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.rnn_lr, weight_decay=cfg.rnn_weight_decay)

    if cfg.rnn_criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()

    num_epochs = cfg.rnn_epochs

    logging.debug(f'''Starting training:
        Epochs:          {cfg.rnn_epochs}
        Batch size:      {cfg.rnn_batch_size}
        Learning rate:   {cfg.rnn_lr}
        Optimizer:       {cfg.rnn_optimizer}
        Criterion:       {cfg.cnn_criterion}
        Training size:   {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Device:          {device.type}
        ''')

    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    for epoch in range(num_epochs):
        logging.debug('Epoch {} / {}'.format(epoch+1, num_epochs))
        net.train()
        corrects = 0
        train_num = 0
        for batch in train_loader:
            batch_images = batch['image']
            batch_labels = batch['label']
            batch_images = torch.FloatTensor(batch_images)
            xdata = batch_images.view(-1, 28, 28)
            output = net(xdata)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss += loss.item()
            corrects += torch.sum(pre_lab == batch_labels.data)
            train_num += batch_images.size(0)
        logging.info('correct prediction : {} / {}'.format(corrects, len(train_loader.dataset)))
        train_loss_all.append(loss / train_num)
        train_acc_all.append(corrects.double().item() / train_num)
        logging.info('Train Loss : {:.4f}  Train Acc: {:.4f}'.format(train_loss_all[-1], train_acc_all[-1]))

        net.eval()
        corrects = 0
        test_num = 0
        for batch in val_loader:
            batch_images = batch['image']
            batch_labels = batch['label']
            xdata = batch_images.view(-1, 28, 28)
            output = net(xdata)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, batch_labels)
            loss += loss.item()
            corrects += torch.sum(pre_lab == batch_labels.data)
            test_num += batch_images.size(0)
        test_loss_all.append(loss / test_num)
        test_acc_all.append(corrects.double().item() / test_num)
        logging.info('Test Loss  : {:.4f}  Test Acc: {:.4f}'.format(test_loss_all[-1], test_acc_all[-1]))

    torch.save(net.state_dict(), model_save_path)
    logging.info('Model parameters have been trained and the Output Path is : {}'.format(model_save_path))

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_all, 'ro-', label='Train')
    plt.plot(test_loss_all, 'bs-', label = "Val loss")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.subplot(1,2,2)
    plt.plot(train_acc_all, 'ro-', label='Train acc')
    plt.plot(test_acc_all, 'bs-', label='Val acc')
    plt.xlabel("epoch")
    plt.ylabel('acc')
    plt.legend()
    plt.show()

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir):
        if not osp.exists(images_dir):
            raise FileNotFoundError('文件目录不存在')
        self.root_dir = osp.join(images_dir)
        image_slice_list = []
        label_list = []
        for image_slice_dir in os.listdir(self.root_dir):
            target_dir = osp.join(self.root_dir, image_slice_dir)
            for image_slice_filename in os.listdir(target_dir):
                slice_path = osp.join(target_dir, image_slice_filename)
                slice_image = Image.open(slice_path).convert('L')
                image_slice_list.append(slice_image)
                label_list.append(image_slice_filename.split('.')[0].split('-')[-1])
        self.image_slice_list = image_slice_list
        self.label_list = label_list

    def __getitem__(self, item):
        img = self.image_slice_list[item]
        label = self.label_list[item]
        np_array_img = np.array(img)
        np_array_img = np.expand_dims(np_array_img, axis=2)
        np_array_img = np_array_img.transpose((2, 0, 1)).astype(float)
        return {'image': torch.from_numpy(np_array_img).float(), 'label': torch.tensor(int(label), dtype=torch.long)}

    def __len__(self):
        return len(self.image_slice_list)

cfg = Config()
logging = LogConfig()

if __name__ == '__main__':
    train_model()