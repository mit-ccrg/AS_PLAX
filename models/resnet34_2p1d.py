import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel


class Resnet34_2p1d_Net(torch.nn.Module):
    """
    Network for 3d Resnet
    """ 

    def __init__(self, opt, t = 3, d = 3):
        super(Resnet34_2p1d_Net, self).__init__()
       

        mid_filters =  (t*d**2*1*64)//(d**2 *1 + t*64)
        self.conv1_1 = nn.Conv3d(1, mid_filters, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1_2  = nn.BatchNorm3d(num_features=mid_filters)
        self.conv2_1 = nn.Conv3d(mid_filters, 64, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0), bias=False)
        self.bn2_2  = nn.BatchNorm3d(num_features=64)

        self.pool1 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2))

        self.stack1 = self.make_stack(64, 64,3, downsample_first=False, max_pool_output=True)
        self.stack2 = self.make_stack(64, 128, 4, downsample_first=True, max_pool_output=True)
        self.stack3 = self.make_stack(128, 256, 6, downsample_first=True, max_pool_output=True)
        self.stack4 = self.make_stack(256, 512, 3, downsample_first=True, max_pool_output=False)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc1 =  nn.Linear(512, 1000)
        self.fc2 = nn.Linear(1000, opt.num_outcomes)

    def make_stack(self, input_conv, output_conv, length, downsample_first= False, max_pool_output= False, stride= 1):
        layers = []

        if downsample_first:
            layers.append(Resnet34_2p1d_Block(input_conv,output_conv, downsample = True, stride= stride ))
        else:
            layers.append(Resnet34_2p1d_Block(input_conv,output_conv, downsample = False, stride= stride ))
        
        for _ in range(length-1):
            layers.append(Resnet34_2p1d_Block(output_conv,output_conv, downsample = False, stride= stride ))

        if max_pool_output:
            layers.append(nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2)))

        return nn.Sequential(*layers)


    def forward(self, x):
        x= self.conv1_1(x)
        x = F.relu(self.bn1_2(x))
        x= self.conv2_1(x)
        x = F.relu(self.bn2_2(x))
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

class Resnet34_2p1d_Block(torch.nn.Module):
    """
    Block for 3d Resnet
    """ 

    def __init__(self, input_filters , output_filters, downsample = False, stride=1, t = 3, d = 3):
        super(Resnet34_2p1d_Block, self).__init__()
       
        mid_filters1 =  (t*d**2*input_filters*output_filters)//((d**2 *input_filters) + (t*output_filters))

        self.conv1_1 = nn.Conv3d(input_filters, mid_filters1, kernel_size=(1,3,3), stride= (1,1,stride), padding=(0,1,1), bias=False)
        self.bn1_1 = nn.BatchNorm3d(num_features=mid_filters1)
        self.conv1_2 = nn.Conv3d(mid_filters1, output_filters, kernel_size=(3,1,1),stride= (stride,1,1), padding=(1,0,0), bias=False)      
        self.bn1_2 = nn.BatchNorm3d(num_features=output_filters)

        mid_filters2 =  (t*d**2*output_filters*output_filters)//((d**2 *output_filters) + (t*output_filters))
        self.conv2_1 = nn.Conv3d(output_filters, mid_filters2, kernel_size=(1,3,3), stride= (1,1,stride), padding=(0,1,1), bias=False)
        self.bn2_1 = nn.BatchNorm3d(num_features=mid_filters2)
        self.conv2_2 = nn.Conv3d(mid_filters2, output_filters, kernel_size=(3,1,1),stride= (stride,1,1), padding=(1,0,0), bias=False)      
        self.bn2_2 = nn.BatchNorm3d(num_features=output_filters)

        
        self.downsample = downsample
        if downsample:
            self.conv3 = nn.Conv3d(input_filters, output_filters, kernel_size=(1,1,1), stride= stride,bias=False )



    def forward(self, x):
        residual = x

        x = self.conv1_1(x)
        x= F.relu(self.bn1_1(x))

        
        x = self.conv1_2(x)
        x= F.relu(self.bn1_2(x))


        x = self.conv2_1(x)
        x= F.relu(self.bn2_1(x))


        x = self.conv2_2(x)
        x= self.bn2_2(x)


        if self.downsample:
            residual = self.conv3(residual)

        x += residual 

        x= F.relu(x)

        return x

class Resnet34_2p1d(BaseModel):
    """
    3d resenet34 Model
    """

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.network = Resnet34_2p1d_Net(opt)
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
