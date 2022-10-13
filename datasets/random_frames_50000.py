import torch.utils.data as data
from abc import ABC, abstractmethod
from utils.utils import loadvideo
import pandas as pd
from .base_dataset import BaseDataset
import os
import numpy as np

class Random_Frames_50000(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.path = opt.data_dir
        self.max_frame = 1
        self.labels = None

    def __len__(self):
        return 50000

    def __getitem__(self, index):
        file_name = os.path.join(self.path, str(index)) + '.npy'
        video_array = np.load(file_name).astype('float32')
        video_array = np.expand_dims(video_array, 0) # add channels dim
        sample={"video":video_array}
        return sample

