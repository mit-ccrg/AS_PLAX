"""
This package contains codes to build training and testing datasets
"""
import torch.utils.data
import importlib
from datasets.base_dataset import BaseDataset

def create_dataset(opt):
    """
    Create a dataset given the options

    Main interface between the datasets package and train/test python scripts
    """

    data_loaders = DatasetLoaders(opt)
    return data_loaders

def find_dataset(dataset_name):
    file_name = "datasets." + dataset_name
    dataset_lib = importlib.import_module(file_name)
    dataset = None
    for name, dataset_class in dataset_lib.__dict__.items():
        if name.lower() == dataset_name.lower() and issubclass(dataset_class, BaseDataset):
            dataset = dataset_class
    if dataset is None:
        raise NotImplementedError("Dataset not found.\n")
    return dataset

class DatasetLoaders():
    def __init__(self, opt):
        self.opt = opt
        dataset_class = find_dataset(opt.dataset_name)
        print("opt.if_split",opt.if_split)
        if opt.if_split:
            self.dataset = dataset_class(opt)
            train_size = int(opt.train_size * len(self.dataset))
            test_size = len(self.dataset) - train_size
            self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = opt.batch_size)
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size = opt.batch_size)
        else: # if train and test/valid set fed seperately
            if opt.is_train == 'testing':
                self.test_dataset = dataset_class(opt, type_data="test")
                self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size = opt.batch_size,shuffle=True,drop_last=True)
                print("---------------Test Set " + opt.dataset_name + " created---------------.\n")
                print("Test Dataset Size: ", len(self.test_dataset))
            else:
                self.train_dataset = dataset_class(opt, type_data="train")
                self.valid_dataset = dataset_class(opt, type_data="valid")
                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size = opt.batch_size,shuffle=True,drop_last=True)
                self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size = opt.batch_size,shuffle=True,drop_last=True)
                print("---------------Dataset " + opt.dataset_name + " created---------------.\n")
                print("Train Dataset Size: ", len(self.train_dataset), 
                    "Valid Dataset Size: ", len(self.valid_dataset))
            

            
            

       

#     def load_data(self):
#         return self

    def __len__(self):
        """
        Return the number of data in the dataset
        """
        return len(self.dataset)

#     def __iter__(self):
#         """
#         Return a batch of data
#         """
#         for data in self.dataloader:
#             yield data


