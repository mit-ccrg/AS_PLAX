import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import BaseModel

class CNN_3d_Net(torch.nn.Module):
    """
    Network for 3d CNN
    """	

    def __init__(self, opt):
        super(CNN_3d_Net, self).__init__()
        self.k_size = tuple([int(i) for i in opt.kernel_size.split('-')])
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.conv4 = nn.Conv3d(128, 128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv5 = nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.pool3 = nn.AdaptiveMaxPool3d((6))
        self.fc1 = nn.Linear(256*6**3, 4096)
        self.bn1 = nn.BatchNorm1d(num_features=4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(num_features=1024)
        self.fc3 = nn.Linear(1024, opt.num_outcomes)

    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
#         print(1,x.shape)
        x = self.pool1(x)
#         print("pool 1",x.shape)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
#         print(2,x.shape)
        x = self.pool2(x)
#         print("pool 2",x.shape)
        x = F.relu(self.conv5(x))
#         print(3,x.shape)
        x = self.pool3(x)
#         print("pool 3",x.shape)
        size = np.prod(list(x.size()[1:]))
        x = x.view(-1, size)
#         print(x.shape)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
#         print(x)
        return x

class CNN_3d(BaseModel):
    """
    Simple 3d CNN
    """

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.network = CNN_3d_Net(opt)
        print('----------------Network Defined----------------\n')
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=opt.lr)


    def forward(self):
        self.output = self.network.forward(self.input_data)

    def backward(self):
     #   loss_function = nn.BCELoss()
        self.loss = self.loss_function(self.output, self.input_label.reshape(self.output.shape))
        print("loss",self.loss, self.output, self.input_label)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad() # set gradients to zero
        self.forward() # compute forward
        self.backward() # compute gradients
        self.optimizer.step() # update weights
    
    def test(self):
     #   loss_function = nn.BCELoss()
        print(self.output, self.input_label)
        self.loss = self.loss_function(self.output, self.input_label.reshape(self.output.shape))