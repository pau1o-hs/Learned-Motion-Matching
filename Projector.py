import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch_optimizer as optim
import random

import numpy as np

projectorIn = []
projectorOut = []

with open("/mnt/c/Users/henri/Documents/Unity Projects/Neural Network/Assets/Motion Matching/Database/Features.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        projectorIn.append(converted)

x = torch.tensor(projectorIn, dtype=torch.float)
# y = torch.tensor(projectorOut, dtype=torch.float)        # x data (tensor), shape=(100, 1)

means = x.mean(dim=1, keepdim=True)
stds = x.std(dim=1, keepdim=True)
normalized_data = (x - means) / stds

torch.set_printoptions(precision=6)
np.set_printoptions(precision=7, floatmode='fixed', suppress=True)

train = Data.TensorDataset(normalized_data, normalized_data)
train_loader = Data.DataLoader(train, shuffle=True, batch_size=32)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)   # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)   # hidden layer
        self.hidden4 = torch.nn.Linear(n_hidden3, n_hidden4)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden4, n_output)   # output layer

    def forward(self, x):
        x = torch.tanh(self.hidden(x))      # activation function for hidden layer
        x = torch.tanh(self.hidden2(x))      # activation function for hidden layer
        x = torch.tanh(self.hidden3(x))      # activation function for hidden layer
        x = torch.tanh(self.hidden4(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=24+32, n_hidden=512, n_hidden2=512, n_hidden3=512, n_hidden4=512, n_output=24+32)     # define the network

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

# train the network
for t in range(50000):
    for batch_idx, (data, target) in enumerate(train_loader):
        noise = torch.randn_like(data)*random.uniform(0, 0.1)
        product = torch.mul(data, noise)
        dataNoise = data + product

        prediction = net(dataNoise)     # input x and predict based on x

        loss = loss_func(prediction, target)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()

prediction = net(normalized_data[0])

print(normalized_data[0].detach().numpy())
print()
print(prediction.detach().numpy())

with open('/mnt/c/Users/henri/Documents/Unity Projects/Neural Network/Assets/Motion Matching/NNWeights/Projector.txt', "w+") as f:
    np.savetxt(f, net.hidden.weight.detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden.bias.detach().numpy(), delimiter="\n")

    np.savetxt(f, net.hidden2.weight.detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden2.bias.detach().numpy(), delimiter="\n")

    np.savetxt(f, net.hidden3.weight.detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden3.bias.detach().numpy(), delimiter="\n")

    np.savetxt(f, net.hidden4.weight.detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden4.bias.detach().numpy(), delimiter="\n")
    
    np.savetxt(f, net.predict.weight.detach().numpy().transpose(), delimiter="\n")
    np.savetxt(f, net.predict.bias.detach().numpy(), delimiter="\n")