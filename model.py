import torch
from torch import nn
# import torch.nn.functional as F

class RatingNet(nn.Module):

	def __init__(self, input_size, hidden_size=128, num_layers=4,
				 batch_first=True, dropout=0, bidirectional=False):
		super(RatingNet, self).__init__()

		self.num_layers = num_layers
		self.input_size = input_size
		self.hidden_size = hidden_size

		# the input is a "state vector" representing the (lat, long) with the speed

		self.lstm = nn.LSTM(input_size=input_size, 
							hidden_size=hidden_size,
							num_layers=num_layers, 
							batch_first=batch_first,
							dropout=dropout,
							bidirectional=bidirectional)

		self.regressor = nn.Linear(hidden_size, 1)

		self.hidden0 = torch.ones(hidden_size)
		self.cell0 = torch.ones(hidden_size)

	def forward(self, trip):

		# hid, cell = self.init_hidden()

		# out, (hid, cell) = self.lstm(trip, hid, cell)
		out, (hid, cell) = self.lstm(trip)

		rating = torch.sigmoid(self.regressor(out[:, -1])).squeeze()

		return rating

	def init_hidden():

		pass

if __name__ == "__main__":

	seq_len = 50
	batch_size = 64
	input_size = 10

	trip = torch.ones(seq_len, batch_size, input_size)
	target = torch.ones(batch_size)

	model = RatingNet(input_size)

	out = model(trip)

	print(len(out))



		