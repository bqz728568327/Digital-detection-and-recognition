import torch

class Lenet(torch.nn.Module):
    def __init__(self, in_channels, cnn_n_classes):
        super(Lenet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 10, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.Dropout(0.5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU()
        )
        self.fc1 = torch.nn.Sequential(
            # 320是根据卷积计算而来4*4*20(4*4表示大小,20表示通道数)
            torch.nn.Linear(320, 50),
            torch.nn.ReLU()
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(50, cnn_n_classes),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # 不确定行 320列
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.classifer(x)
        return x