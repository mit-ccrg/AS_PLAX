import torch.utils.data as data
from abc import ABC, abstractmethod

class BaseDataset(data.Dataset, ABC):
	def __init__(self, opt):
		self.opt = opt

	@abstractmethod
	def __len__(self):
		return 0

	@abstractmethod
	def __getitem__(self, index):
		pass

	def __scale__(self):
		# implements scaling function
		pass
