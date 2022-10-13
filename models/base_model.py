import os
import torch
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    An abstract base class for all models
    """
    def __init__(self, opt):
        self.opt =  opt
        self.set_loss()
        # extract a few useful options here?

    @abstractmethod
    def forward(self):
        """ 
        Run forward pass
        """
        pass

    @abstractmethod
    def optimize_parameters(self):
        """
        Calculate loss, gradients and update network weights.
        """
        pass
    
    @abstractmethod
    def test(self):
        """
        Calculate loss during testing
        """
        pass

    def set_input(self, data):      
        device = self.opt.device   
        self.input_data = data['video'].to(device)
        if 'outcome' in data:
            self.input_label = data['outcome'].to(device)
        if 'mask' in data:
            self.mask = data['mask'].to(device)

    def get_current_losses(self):
        return self.loss

    def set_loss(self):
        loss_function = self.opt.loss_function
        if loss_function == "BCE":
            self.loss_function = torch.nn.BCELoss()
        elif loss_function == "MSE":
            self.loss_function = torch.nn.MSELoss()

    def save_model(self, epoch):
        """
        save models to disk
        """
        check_point_dir = self.opt.save_dir + '_' + self.opt.date + '/'
        path = os.path.join(check_point_dir, "checkpoint_epoch_"+str(epoch))
        torch.save(self.network.state_dict(), path)

    def load_model(self, epoch=None, path=None):
        """
        load models from disk
        """
#         if not path:
#             if not epoch:
#                 paths = os.listdir(self.opt.save_dir)
#                 #finds the path with the largest epoch
#                 paths.sort(key=lambda x: int(x.split("_")[-1]))
#                 path = os.path.join(self.opt.save_dir, paths[-1])
#             if epoch:
#                 path = os.path.join(self.opt.save_dir, "checkpoint_epoch_"+str(epoch))
        self.network.load_state_dict(torch.load(path))             

    def print_model(self):
        """
        print model structure and weights
        """
        pass
