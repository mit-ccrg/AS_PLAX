import torch.utils.data as data
from abc import ABC, abstractmethod
from utils.utils import loadvideo
import pandas as pd
from .base_dataset import BaseDataset
import os
import numpy as np
import torchvideo.transforms as VT

class Dataset_Example(BaseDataset):
    
    def __init__(self, opt, type_data = "train"):
        np.random.seed(10)
        BaseDataset.__init__(self, opt)
        self.data_path = opt.data_dir
        self.label_path = opt.label_dir
        self.label_file = opt.label_file
        self.view = opt.view
        self.view_model = opt.view_model
        self.num_frames = opt.num_frames
        self.masked = opt.masked
        self.ecg_aligned = opt.ecg_aligned
        if type_data=="train":
            self.labels = pd.read_csv(os.path.join(self.label_path, self.label_file + '_train.csv'))
        elif type_data == "valid":
            self.labels = pd.read_csv(os.path.join(self.label_path, self.label_file + '_valid.csv'))
        else:
            self.labels = pd.read_csv(os.path.join(self.label_path, self.label_file + '_test.csv'))
        self.input_size = opt.input_size
        self.size = [int(i) for i in opt.input_size.split("-")]
        self.transform = opt.transform
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        file_name = os.path.join(self.data_path, self.labels.iloc[index]['vid_dir'])
        video_array = np.load(file_name)/255.0
        video_frames = video_array.shape[0]
        if self.ecg_aligned == 'True':
            start_frame = 0
        else:
            start_frame = np.random.randint(0, video_frames - self.num_frames +1)
        video_array = video_array[start_frame:start_frame+self.num_frames, :, :]
        video_array = video_array.reshape(1, self.num_frames, video_array.shape[1], video_array.shape[2])
       
        outcome = np.array(self.labels.iloc[index]['out_come']).astype('float32')
        #apply transform
        if self.transform:
            resize_factor = np.random.uniform(1.0,1.25)
            transform = VT.Compose([
            VT.NDArrayToPILVideo(),
            VT.ResizeVideo((int(self.size[0]*resize_factor),int(self.size[1]*resize_factor))),
            VT.RandomCropVideo(self.size,pad_if_needed=True),
            VT.CollectFrames(),
            VT.PILVideoToTensor()
        ])
            video_array = transform(video_array[0])
        sample={"file_name":file_name, "video":video_array, 'outcome':outcome}
        return sample