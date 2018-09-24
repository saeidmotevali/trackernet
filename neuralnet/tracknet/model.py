import torch.nn.functional as F
from torch import nn


class TrackNet(nn.Module):
    def __init__(self, width, channels):
        super(TrackNet, self).__init__()

        self.channels = channels
        self.width = width

        self.kern_size = 3
        self.kern_stride = 1
        self.kern_padding = 1
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool1 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv1 = nn.Conv2d(self.channels, 64, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()
        self.bn1 = nn.BatchNorm2d(64)

        self.kern_size = 3
        self.kern_stride = 2
        self.kern_padding = 1
        self.mxp_kern_size = 2
        self.mxp_stride = 2
        self.pool2 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv2 = nn.Conv2d(64, 256, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()
        self.bn2 = nn.BatchNorm2d(256)

        self.kern_size = 3
        self.kern_stride = 1
        self.kern_padding = 1
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool3 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv3 = nn.Conv2d(256, 256, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()
        self.bn3 = nn.BatchNorm2d(256)

        self.kern_size = 3
        self.kern_stride = 1
        self.kern_padding = 1
        self.mxp_kern_size = 2
        self.mxp_stride = 2
        self.pool4 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv4 = nn.Conv2d(256, 512, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()
        self.bn4 = nn.BatchNorm2d(512)

        self.kern_size = 3
        self.kern_stride = 1
        self.kern_padding = 1
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool5 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv5 = nn.Conv2d(512, 512, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()
        self.bn5 = nn.BatchNorm2d(512)

        self.kern_size = 1
        self.kern_stride = 1
        self.kern_padding = 0
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool6 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv6 = nn.Conv2d(512, 256, self.kern_size,
                                stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()
        self.bn6 = nn.BatchNorm2d(256)

        self.linearWidth = 256 * int(self.width) * int(self.width)
        self.fc1 = nn.Linear(self.linearWidth, 128)
        self.out = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.pool6(F.relu(self.bn6(self.conv6(x))))

        x = x.view(-1, self.linearWidth)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

    def _update_output_size(self):
        temp = self.width
        self.width = ((self.width - self.kern_size + 2 * self.kern_padding) / self.kern_stride) + 1
        temp1 = self.width
        self.width = ((self.width - self.mxp_kern_size) / self.mxp_stride) + 1
        print('Output width[ ' + str(temp) + ' -conv-> ' + str(temp1) + ' -maxpool-> ' + str(self.width) + ' ]')
        self.width = int(self.width)

for i in [15, 21]:
    print(i, "######################################################")
    k = TrackNet(i, 1)
