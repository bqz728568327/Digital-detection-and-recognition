# -*- coding: utf-8 -*-
# @Time    : 2021-11-11
# @Author  : Bai
# @Email   : 728568327@qq.com
# @File    : train_model_cnn.py

import torch
import os.path as osp
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from config_parameter import Config
from config_logging import LogConfig
from model.cnn import Lenet

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_dir = cfg.train_slice_image_root
    model_save_path = cfg.cnn_model_save_path

    logging.debug(f'''Network::
    points:          {cfg.cnn_model}
    input channels:  {cfg.cnn_in_channels}
    output channels: {cfg.cnn_n_classes}
    ''')

    dataset = ImageDataset(images_dir)
    val_percent = cfg.cnn_val_percent

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train,
                              batch_size=cfg.cnn_batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)
    val_loader = DataLoader(val,
                            batch_size=cfg.cnn_batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True)

    net = Lenet(cfg.cnn_in_channels, cfg.cnn_n_classes)

    if cfg.cnn_optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.cnn_lr, weight_decay=cfg.cnn_weight_decay)

    if cfg.cnn_criterion == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()

    num_epochs = cfg.cnn_epochs

    logging.debug(f'''Starting training:
    Epochs:          {cfg.cnn_epochs}
    Batch size:      {cfg.cnn_batch_size}
    Learning rate:   {cfg.cnn_lr}
    Optimizer:       {cfg.cnn_optimizer}
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
            output = net(batch_images)
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
        logging.info('Train Loss: {:.4f}  Train Acc: {:.4f}'.format(train_loss_all[-1], train_acc_all[-1]))
        net.eval()
        corrects = 0
        test_num = 0
        for batch in val_loader:
            batch_images = batch['image']
            batch_labels = batch['label']
            output = net(batch_images)
            pre_lab = torch.argmax(output, 1)
            loss = criterion(output, batch_labels)
            loss += loss.item()
            corrects += torch.sum(pre_lab == batch_labels.data)
            test_num += batch_images.size(0)
        test_loss_all.append(loss / test_num)
        test_acc_all.append(corrects.double().item() / test_num)
        logging.info('Test Loss: {:.4f}  Test Acc: {:.4f}'.format(test_loss_all[-1], test_acc_all[-1]))

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