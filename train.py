import torch
from torch import nn
from torch import optim

def train(model, dataloader, n_epochs=4, lr=1e-2):

	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)

	for epoch in range(n_epochs):

		for batch_idx, (inputs, targets) in enumerate(dataloader):

			optimizer.zero_grad()

			outputs = model(inputs)

			loss = criterion(outputs, targets)

			loss.backward()
			optimizer.step()


			print(loss.data)

if __name__ == "__main__":

	from model import RatingNet

	seq_len = 50
	batch_size = 64
	input_size = 10

	trip = torch.ones(batch_size, seq_len, input_size)
	target = torch.ones(batch_size)

	model = RatingNet(input_size)

	from torch.utils.data import TensorDataset, DataLoader

	toy_dataset = TensorDataset(trip, target)
	toy_dataloader = DataLoader(toy_dataset, batch_size=batch_size)

	train(model, toy_dataloader, n_epochs=3)

	import sys
	print(sys.getsizeof(model))