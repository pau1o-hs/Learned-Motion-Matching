from numpy.random import random_integers
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
import torch_optimizer as optim
import random

import numpy as np
import time

start_time = time.time()

stepperInput = []
stepperNext = []

with open("/home/pau1o-hs/Documents/Database/Features.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        stepperInput.append(converted)

with open("/home/pau1o-hs/Documents/Database/Features.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        stepperNext.append(converted)

# stepperNext = np.roll(stepperInput, -1, axis=0)

x = torch.tensor(stepperInput, dtype=torch.float)        # x data (tensor), shape=(100, 1)
y = torch.tensor(stepperNext,  dtype=torch.float)

# means = x.mean(dim=1, keepdim=True)
# stds = x.std(dim=1, keepdim=True)
# normalized_input = (x - means) / stds

# means = y.mean(dim=1, keepdim=True)
# stds = y.std(dim=1, keepdim=True)
# normalized_output = (y - means) / stds

# torch can only train on Variable, so convert them to Variable
# x, y = Variable(x), Variable(y)

class MyDataset(Dataset):
    def __init__(self, data, window=20):
        self.data = data
        self.window = window

    def __getitem__(self, index):
        x = self.data[index:index+self.window]
        return x

    def __len__(self):
        return len(self.data) - self.window

train = Data.TensorDataset(x, y)

dataSet = MyDataset(train)

train_loader = Data.DataLoader(dataSet, shuffle=True, batch_size=20)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        x = torch.tanh(self.hidden(x))      # activation function for hidden layer
        x = torch.tanh(self.hidden2(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=24+32, n_hidden=512, n_hidden2=512, n_output=24+32)     # define the network

# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# train the network
for t in range(1000):
    for batch_idx, (data, target) in enumerate(train_loader):    
        prediction = net(data)     # input x and predict based on x

        loss = loss_func(prediction, target)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()

torch.set_printoptions(precision=6)
np.set_printoptions(precision=7, floatmode='fixed', suppress=True)

prediction = net(x[0])

print(x[0].detach().numpy())
print()
print(y[0].detach().numpy())
print()
print(prediction.detach().numpy())

with open('/mnt/c/Users/henri/Documents/Unity Projects/Neural Network/Assets/Motion Matching/NNWeights/Stepper.txt', "w+") as f:
    np.savetxt(f, net.hidden.weight.detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden.bias.detach().numpy(), delimiter="\n")

    np.savetxt(f, net.hidden2.weight.detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden2.bias.detach().numpy(), delimiter="\n")

    np.savetxt(f, net.predict.weight.detach().numpy().transpose(), delimiter="\n")
    np.savetxt(f, net.predict.bias.detach().numpy(), delimiter="\n")

print("Runtime: %s minutes" % ((time.time() - start_time) / 60))