import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel


class Resnet34_mc3_Net(torch.nn.Module):
    """
    Network for 3d Resnet
    """ 

    def __init__(self, opt):
        super(Resnet34_mc3_Net, self).__init__()
       
        # seperate networks for two outcomes
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1  = nn.BatchNorm3d(num_features=64)
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

        self.stack1 = self.make_stack(64, 64,3, downsample_first=False, max_pool_output=True)
        self.stack2 = self.make_stack(64, 128, 4, downsample_first=True, max_pool_output=True, padding=(0,1,1), kernel_size=(1,3,3))
        self.stack3 = self.make_stack(128, 256, 6, downsample_first=True, max_pool_output=True, padding=(0,1,1),kernel_size=(1,3,3))
        self.stack4 = self.make_stack(256, 512, 3, downsample_first=True, max_pool_output=False, padding=(0,1,1),kernel_size=(1,3,3))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 =  nn.Linear(512, 1000)
        self.fc2 = nn.Linear(1000, opt.num_outcomes)

    def make_stack(self, input_conv, output_conv, length, downsample_first= False, max_pool_output= False, stride= 1, kernel_size=3, padding=1):
        layers = []
        if downsample_first:
            layers.append(Resnet34_mc3_Block(input_conv,output_conv, downsample = True,
             kernel_size=kernel_size,stride= stride, padding=padding ))
        else:
            layers.append(Resnet34_mc3_Block(input_conv,output_conv, downsample = False,
             kernel_size=kernel_size, stride= stride,padding=padding ))
        
        for _ in range(length-1):
            layers.append(Resnet34_mc3_Block(output_conv,output_conv, downsample = False,
             kernel_size= kernel_size,stride= stride,padding=padding ))

        if max_pool_output:
            layers.append(nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2)))

        return nn.Sequential(*layers)


    def forward(self, x):
        x= self.conv1(x)
        x = F.relu(self.bn1(x))
        x= self.pool1(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        x= self.avgpool(x)
        x= torch.reshape(x, (-1, 512))
        x= self.fc1(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class Resnet34_mc3_Block(torch.nn.Module):
    """
    Block for 3d Resnet
    """ 

    def __init__(self, input_filters , output_filters, downsample = False, stride=1, kernel_size=(3,3,3), padding= 1):
        super(Resnet34_mc3_Block, self).__init__()
       
        self.conv1 = nn.Conv3d(input_filters, output_filters, kernel_size=kernel_size, stride= stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm3d(num_features=output_filters)
        self.conv2 = nn.Conv3d(output_filters, output_filters, kernel_size=kernel_size,stride= stride, padding=padding, bias=False)      
        self.bn2 = nn.BatchNorm3d(num_features=output_filters)
        
        self.downsample = downsample
        if downsample:
            self.conv3 = nn.Conv3d(input_filters, output_filters, kernel_size=(1,1,1), stride= stride,bias=False )



    def forward(self, x):
        residual = x


        x = self.conv1(x)
        x= F.relu(self.bn1(x))


        x = self.conv2(x)
        x= self.bn2(x)


        if self.downsample:
            residual = self.conv3(residual)

        x += residual 

        x= F.relu(x)

        return x

class Resnet34_mc3(BaseModel):
    """
    3d resenet34 Model
    """

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.network = Resnet34_mc3_Net(opt)
        print('----------------Network Defined----------------\n')
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    def forward(self):
        self.output = self.network.forward(self.input_data)

    def backward(self):
        self.loss = self.loss_function(self.output, self.input_label)
        print(self.loss)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad() # set gradients to zero
        self.forward() # compute forward
        self.backward() # compute gradients
        self.optimizer.step() # update weights
    
    def test(self):
        print(self.output, self.input_label)
        self.loss = self.loss_function(self.output, self.input_label.reshape(self.output.shape))
