"""
This package contains models to do classification
"""
import importlib
from models.base_model import BaseModel
import torch


def find_model(model_type):
    file_name = "models." + model_type
    model_lib = importlib.import_module(file_name)
    model = None
    for name, model_class in model_lib.__dict__.items():
        if name.lower() == model_type.lower() and issubclass(model_class, BaseModel):
            model = model_class
    if not model:
        print("Could not find model", model_type)
    return model

def create_model(opt):
    """
    Create a model given the options
    """
    model_type = opt.model # get name of the model
    device = opt.device # get model device
    model_class = find_model(model_type) # get the class corresponding to name of the model type 
    model_instance = model_class(opt)
    print('----------------Model Created----------------\n')
#     model_instance.network = torch.nn.parallel.DistributedDataParallel(model_instance.network)
    model_instance.network = torch.nn.DataParallel(model_instance.network)
    model_instance.network.to(device)
    pytorch_total_params = sum(p.numel() for p in model_instance.network.parameters())
    pytorch_total_params = sum(p.numel() for p in model_instance.network.parameters() if p.requires_grad)
    if opt.verbose > 0:
        print("total params", pytorch_total_params)
        print("learnable params", pytorch_total_params)
    if opt.verbose > 1:
        print(model_instance.network)
    return model_instance