import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel

class Pre_Train_CNN_Net(torch.nn.Module):
    """
    Network for 3d CNN
    """ 

    def __init__(self, opt):
        super(Pre_Train_CNN_Net, self).__init__()
#         self.k_size = tuple([int(i) for i in opt.kernel_size.split('-')])
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.conv2 = nn.Conv3d(16, 16, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv3 = nn.Conv3d(16, 32, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
        self.conv4 = nn.Conv3d(32, 32, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.pool3 = nn.AdaptiveMaxPool3d((6)) #6
        if opt.masked:
            self.fc1 = nn.Linear(32*6**3 + opt.num_frames, 512)
        else:
            self.fc1 = nn.Linear(32*6**3, 512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(num_features=128)
       
        # seperate networks for two outcomes
        self.fc3_1 = nn.Linear(128, 16)
        self.bn3_1 = nn.BatchNorm1d(num_features=16)
        self.fc4_1 = nn.Linear(16, opt.num_outcomes)

        self.fc3_2 = nn.Linear(128, 16)
        self.bn3_2 = nn.BatchNorm1d(num_features=16)
        self.fc4_2 = nn.Linear(16, opt.num_outcomes)
#         self.fc1 = nn.Linear(32*6**3, 1024)
#         self.bn1 = nn.BatchNorm1d(num_features=1024)
#         self.fc2 = nn.Linear(1024, 256)
#         self.bn2 = nn.BatchNorm1d(num_features=256)
#         # seperate networks for two outcomes
#         self.fc3_1 = nn.Linear(256, 32)
#         self.bn3_1 = nn.BatchNorm1d(num_features=32)
#         self.fc4_1 = nn.Linear(32, 1)

#         self.fc3_2 = nn.Linear(256, 32)
#         self.bn3_2 = nn.BatchNorm1d(num_features=32)
#         self.fc4_2 = nn.Linear(32, 1)

#         self.conv1 = nn.Conv3d(1, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
#         self.conv2 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
#         self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
#         self.conv3 = nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1))
#         self.conv4 = nn.Conv3d(128, 128, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
#         self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
#         self.conv5 = nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
#         self.conv6 = nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
#         self.pool3 = nn.AdaptiveMaxPool3d((6)) #6
#         self.fc1 = nn.Linear(64*4**3, 2048)
#         self.bn1 = nn.BatchNorm1d(num_features=2048)
#         self.fc2 = nn.Linear(2048, 512)
#         self.bn2 = nn.BatchNorm1d(num_features=512)
#         # seperate networks for two outcomes
#         self.fc3_1 = nn.Linear(512, 128)
#         self.bn3_1 = nn.BatchNorm1d(num_features=128)
#         self.fc4_1 = nn.Linear(128, 1)

#         self.fc3_2 = nn.Linear(512, 128)
#         self.bn3_2 = nn.BatchNorm1d(num_features=128)
#         self.fc4_2 = nn.Linear(128, 1)

#         self.fc1 = nn.Linear(128*6**3, 4096)
#         self.bn1 = nn.BatchNorm1d(num_features=4096)
#         self.fc2 = nn.Linear(4096, 1024)
#         self.bn2 = nn.BatchNorm1d(num_features=1024)
#         # seperate networks for two outcomes
#         self.fc3_1 = nn.Linear(1024, 512)
#         self.bn3_1 = nn.BatchNorm1d(num_features=512)
#         self.fc4_1 = nn.Linear(512, 1)

#         self.fc3_2 = nn.Linear(1024, 512)
#         self.bn3_2 = nn.BatchNorm1d(num_features=512)
#         self.fc4_2 = nn.Linear(512, 1)


    def forward(self, x, mask=None):
#         print(x.shape)
        x = F.relu(self.conv1(x.float()))
        x = F.relu(self.conv2(x))
#         print(1,x.shape)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
#         x = self.pool2(x)
#         x = F.relu(self.conv5(x))
#         x = F.relu(self.conv6(x))
        x = self.pool3(x)
#         print("pool 3",x.shape)
        size = np.prod(list(x.size()[1:]))
        x = x.view(-1, size)
#         print(x.shape)
#         print("before",x.shape,mask.shape)
        if mask is not None:
            x = torch.cat((x, mask.float()), dim=1)
#         print("after",x.shape)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        x1 = F.relu(self.bn3_1(self.fc3_1(x)))
#         print(x1.tolist())
        x1 = torch.sigmoid(self.fc4_1(x1))

        x2 = F.relu(self.bn3_2(self.fc3_2(x)))
        x2 = torch.sigmoid(self.fc4_2(x2))
#         x1=self.fc4_1(x1)
#         x2=self.fc4_2(x2)

        return x1, x2

class Pre_Train_CNN(BaseModel):
    """
    Simple 3d CNN
    """

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.network = Pre_Train_CNN_Net(opt)
        self.masked = opt.masked
        print('----------------Network Defined----------------\n')
        if opt.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        if opt.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)


    def forward(self):
        if self.masked:
            self.output1, self.output2 = self.network.forward(self.input_data, self.mask)
        else:
            self.output1, self.output2 = self.network.forward(self.input_data)
#         self.output1 = self.network.forward(self.input_data)


    def backward(self):
    #    loss_function = nn.BCELoss() #nn.BCEWithLogitsLoss() #nn.BCELoss()
     
        self.loss1 = self.loss_function(self.output1, self.input_label[:,0].reshape(self.output1.shape))
        self.loss2 = self.loss_function(self.output2, self.input_label[:,1].reshape(self.output2.shape))
        self.loss = [self.loss1, self.loss2]
#         print("loss",self.loss1, self.loss2)
        self.total_loss = self.loss1 + self.loss2
#         print("loss",self.loss, self.output, self.input_label)
        self.total_loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad() # set gradients to zero
        self.forward() # compute forward
        self.backward() # compute gradients
        self.optimizer.step() # update weights
    
    def test(self):
  #      loss_function = nn.BCELoss()
#         print(self.output, self.input_label)
        self.loss1 = self.loss_function(self.output1, self.input_label[:,0].reshape(self.output1.shape))
        self.loss2 = self.loss_function(self.output2, self.input_label[:,1].reshape(self.output2.shape))
        self.loss = [self.loss1, self.loss2]