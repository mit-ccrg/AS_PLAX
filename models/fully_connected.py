import torch
import torch.nn.functional as F
from .base_model import BaseModel

class Fully_Connected_Net(torch.nn.Module):
	"""
	Network for Simple fully connected
	"""	

	def __init__(self, opt):
		super(Fully_Connected_Net, self).__init__()
		self.fc1 = torch.nn.Linear(opt.input_size, 128)
		self.fc2 = torch.nn.Linear(128, 64)
		self.fc3 = torch.nn.Linear(64, opt.num_outcomes)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.sigmoid(self.fc3(x))
		return x

class Fully_Connected(BaseModel):
	"""
	Simple fully connected
	"""

	def __init__(self, opt):
		BaseModel.__init__(self, opt)
		self.network = Fully_Connected_Net(opt)
		print('----------------Network Defined----------------\n')
		self.optimizer = torch.optim.Adam(self.network.parameters(), lr=opt.lr)


	def forward(self):
		self.output = self.network.forward(self.input_data)

	def backward(self):
	#	loss_function = torch.nn.BCELoss()
		self.loss = self.loss_function(self.output, self.input_label)
		self.loss.backward()

	def optimize_parameters(self):
		self.optimizer.zero_grad() # set gradients to zero
		self.forward() # compute forward
		self.backward() # compute gradients
		self.optimizer.step() # update weights

	def test(self):
	#	loss_function = torch.nn.BCELoss()
		print(self.output, self.input_label)
		self.loss = self.loss_function(self.output, self.input_label.reshape(self.output.shape))
