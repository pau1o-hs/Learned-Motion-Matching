from os import truncate
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch_optimizer as optim

import numpy as np
import time

start_time = time.time()

decompressorIn = []
decompressorOut = []

with open("/home/pau1o-hs/Documents/Database/Features.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        decompressorIn.append(converted)

with open("/home/pau1o-hs/Documents/Database/Poses.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        decompressorOut.append(converted)

device = torch.device("cuda")

x = torch.tensor(decompressorIn,  dtype=torch.float).to(device)        # x data (tensor), shape=(100, 1)
y = torch.tensor(decompressorOut, dtype=torch.float).to(device)

means = x.mean(dim=0, keepdim=True)
stds = x.std(dim=0, keepdim=True)
normalized_data = (x - means) / (stds + 1e-11)

print(stds)

# torch can only train on Variable, so convert them to Variable
# x, y = Variable(x), Variable(y)
train = Data.TensorDataset(normalized_data, y)
train_loader = Data.DataLoader(train, shuffle=True, batch_size=32)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = torch.tanh(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=24+32, n_hidden=512, n_output=518).to(device)     # define the network

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

torch.set_printoptions(precision=6)
np.set_printoptions(precision=7, floatmode='fixed', suppress=True)

# train the network
for t in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):

        data.to(device), target.to(device)

        prediction = net(data)     # input x and predict based on x

        loss = loss_func(prediction, target)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()

        print(loss.item() * 1000)


prediction = net(normalized_data[0])

# print(x[0].cpu().detach().numpy())
# print()
# print(y[0].cpu().detach().numpy())
# print()
# print(prediction.cpu().detach().numpy())

with open('/home/pau1o-hs/Documents/NNWeights/Decompressor.txt', "w+") as f:
    np.savetxt(f, net.hidden.weight.cpu().detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden.bias.cpu().detach().numpy(), delimiter="\n")

    np.savetxt(f, net.predict.weight.cpu().detach().numpy().transpose(), delimiter="\n")
    np.savetxt(f, net.predict.bias.cpu().detach().numpy(), delimiter="\n")

print("Runtime: %s minutes" % ((time.time() - start_time) / 60))