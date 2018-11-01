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
        
class WindowMLP(nn.Module()):

    def __init__(self, input_size, output_size, hidden_size=1000):
        super(WindowMLP, self).__init__()
        
        # for now, we only only use a model with 2 hidden layers of the same size
        
        # we want to go n steps in the future and n steps in the past, plus the present, so always odd
        if input_size % 2 != 0:
            raise Exception("Size of input must be odd, not even")
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class WindowCNN(nn.Module()):
    
    def __init__(self, input_size, output_size, n_features=6):
        # this model may be more appropriate since CNNs can take into account the spatial structure of the input
        super(WindowCNN, self).__init__()

        # we want to go n steps in the future and n steps in the past, plus the present, so always odd
        if input_size % 2 != 0:
            raise Exception("Size of input must be odd, not even")
        
        self.conv1 = nn.Conv1d()
        
        self.clf = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        
        
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        x = 
        
class TripLSTM(nn.Module()):
    
    # this network takes into account the whole trip to make a decision.
    
    def __init__(self, n_features=6, output_size=6, hidden_size=64, n_layers=1, bidirectional=False):
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            n_layers=n_layers, bidirectional=bidirectional)
        
        #linear layer used to classify hidden states outputted by the LSTM network at each timestep
        self.clf = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        
        x, (self.cell, self.hidden) = self.lstm(x, (self.cell, self.hidden))
        
        # may not be necessary
        x = F.relu(x)
        
        x = self.clf(x)
        
        return x
    
if __name__ == "__main__":

	seq_len = 50
	batch_size = 64
	input_size = 10

	trip = torch.ones(seq_len, batch_size, input_size)
	target = torch.ones(batch_size)

	model = RatingNet(input_size)

	out = model(trip)

	print(len(out))



		