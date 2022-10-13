import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel
import torchvision


class Pre_Train_Res(BaseModel):
    """
    Simple 3d CNN
    """

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.network = torchvision.models.video.r2plus1d_18(pretrained=False)
        self.network.fc = nn.Linear(self.network.fc.in_features, opt.num_outcomes)
        self.network.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        print('----------------Network Defined----------------\n')
        if opt.is_train == "training":
            if opt.optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            if opt.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.network.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)


    def forward(self):
        output = self.network.forward(self.input_data.float())

        loss_function = self.opt.loss_function
        if loss_function == "BCE":
            self.output1 = torch.sigmoid(output)
        elif loss_function == "MSE":
            self.output1 = output
#         self.output1 = torch.sigmoid(output[:,0])
#         self.output2 = torch.sigmoid(output[:,1])
#         self.output1 = self.network.forward(self.input_data)


    def backward(self):
    #    loss_function = nn.BCELoss() #nn.BCEWithLogitsLoss() #nn.BCELoss()
     
        self.loss1 = self.loss_function(self.output1, self.input_label.reshape(self.output1.shape))
#         self.loss2 = loss_function(self.output2, self.input_label[:,1].reshape(self.output2.shape))
        self.loss = [self.loss1, None]#self.loss2]
        self.total_loss = self.loss1# + self.loss2
        self.total_loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad() # set gradients to zero
        self.forward() # compute forward
        self.backward() # compute gradients
        self.optimizer.step() # update weights
    
    def test(self):
 #       loss_function = nn.BCELoss()
#         print(self.output, self.input_label)
        self.loss1 = self.loss_function(self.output1, self.input_label.reshape(self.output1.shape))
#         self.loss2 = loss_function(self.output2, self.input_label[:,1].reshape(self.output2.shape))
        self.loss = [self.loss1, None]# self.loss2]