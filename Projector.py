import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch_optimizer as optim
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import time

start_time = time.time()

projectorIn = []
projectorOut = []
latent = []

with open("/home/pau1o-hs/Documents/Database/XData.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]

        if inner_list == ['']:
            continue

        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        projectorIn.append(converted)

with open("/home/pau1o-hs/Documents/Database/LatentVariables.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        
        if inner_list == ['']:
            continue
        
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        latent.append(converted)

device = torch.device("cuda")

projectorOut = np.append(projectorIn, latent, axis=1)
x = torch.tensor(projectorIn, dtype=torch.float).to(device)
y = torch.tensor(projectorOut, dtype=torch.float).to(device)

means = x.mean(dim=1, keepdim=True)
stds = x.std(dim=1, keepdim=True)
normalized_in = (x - means) / stds

means = y.mean(dim=1, keepdim=True)
stds = y.std(dim=1, keepdim=True)
normalized_out = (y - means) / stds

torch.set_printoptions(precision=6)
np.set_printoptions(precision=7, floatmode='fixed', suppress=True)

train = Data.TensorDataset(normalized_in, normalized_out)
train_loader = Data.DataLoader(train, shuffle=True, batch_size=32)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)    # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)   # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)  # hidden layer
        self.hidden4 = torch.nn.Linear(n_hidden3, n_hidden4)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden4, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))       # activation function for hidden layer
        x = F.relu(self.hidden2(x))      # activation function for hidden layer
        x = F.relu(self.hidden3(x))      # activation function for hidden layer
        x = F.relu(self.hidden4(x))      # activation function for hidden layer
        x = self.predict(x)              # linear output
        return x

def my_loss(noiseInput, predict, target):
    l2loss = torch.abs((noiseInput - target[:, :24].clone())**2 - (noiseInput - predict[:, :24].clone())**2)
    loss = torch.mean(torch.abs(target - predict) + torch.cat((l2loss, torch.zeros(noiseInput.size(0), 64).to(device)), 1))
    return loss

net = Net(n_feature=24, n_hidden=512, n_hidden2=512, n_hidden3=512, n_hidden4=512, n_output=24+64).to(device)     # define the network

optimizer = optim.RAdam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
loss_func = torch.nn.MSELoss() # this is for regression mean squared loss

writer = SummaryWriter()

# train the network
for t in range(10000):
    epochLoss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):

        data.to(device), target.to(device)

        noise = torch.randn(data.size(0), 24).to(device)*(random.uniform(0.0, 1.0)**(0.5))
        dataNoise = data + noise

        knnIndices = []
        newTargets = torch.zeros(target.size(0), target.size(1)).to(device)

        for i in range(0, len(data)):
            dist = torch.norm(normalized_in - dataNoise[i], dim=1, p=None)
            knn = dist.topk(1, largest=False)
            knnIndices.append(knn.indices)

        for i in range(0, len(data)):
            newTargets[i] = normalized_out[knnIndices[i]]

        prediction = net(dataNoise)     # input x and predict based on x

        loss = loss_func(prediction, newTargets)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()

        epochLoss += loss * prediction.size(0)

    if t % 500 == 0:
        print(t, epochLoss.item())
    
    writer.add_scalar('Python Projector Loss', epochLoss, t)

# prediction = net(normalized_data[0])

# print(normalized_data[0].detach().numpy())
# print()
# print(prediction.detach().numpy())

with open('/home/pau1o-hs/Documents/NNWeights/Projector.txt', "w+") as f:
    np.savetxt(f, net.hidden.weight.cpu().detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden.bias.cpu().detach().numpy(), delimiter="\n")

    np.savetxt(f, net.hidden2.weight.cpu().detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden2.bias.cpu().detach().numpy(), delimiter="\n")

    np.savetxt(f, net.hidden3.weight.cpu().detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden3.bias.cpu().detach().numpy(), delimiter="\n")

    np.savetxt(f, net.hidden4.weight.cpu().detach().numpy().transpose(), delimiter="\n")    
    np.savetxt(f, net.hidden4.bias.cpu().detach().numpy(), delimiter="\n")
    
    np.savetxt(f, net.predict.weight.cpu().detach().numpy().transpose(), delimiter="\n")
    np.savetxt(f, net.predict.bias.cpu().detach().numpy(), delimiter="\n")

print("Runtime: %s minutes" % ((time.time() - start_time) / 60))