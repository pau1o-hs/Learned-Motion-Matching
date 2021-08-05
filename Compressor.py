import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch_optimizer as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

compressorIn = []

with open("/home/pau1o-hs/Documents/Database/Poses.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]

        if inner_list == ['']:
            continue

        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        compressorIn.append(converted)

# print(torch.cuda.get_device_name(0)) 

# device = torch.device("cuda")
device = torch.device("cpu")

x = torch.tensor(compressorIn, dtype=torch.float).to(device)        # x data (tensor), shape=(100, 1)
y = torch.tensor(compressorIn, dtype=torch.float).to(device)

# torch can only train on Variable, so convert them to Variable
# x, y = Variable(x), Variable(y)

means = x.mean(dim=1, keepdim=True)
stds  = x.std(dim=1, keepdim=True)
normalized_data = (x - means) / stds

# print(means)
# print(means.shape)
# print(stds)
# print(stds.shape)
# print(normalized_data)

train = Data.TensorDataset(normalized_data, normalized_data)
train_loader = Data.DataLoader(train, shuffle=True, batch_size=32)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)   # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden3, n_output)   # output layer

    def forward(self, x):
        x = torch.tanh(self.hidden(x))      # activation function for hidden layer
        x = torch.tanh(self.hidden2(x))      # activation function for hidden layer
        x = torch.tanh(self.hidden3(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

print(x.shape)

net = Net(n_feature=539, n_hidden=512, n_hidden2=31, n_hidden3=512, n_output=539).to(device)     # define the network
print(next(net.parameters()).is_cuda)

optimizer = torch.optim.Adamax(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

writer = SummaryWriter()

# train the network
for t in range(1000):
    epochLoss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        
        data.to(device), target.to(device)
        
        prediction = net(data)     # input x and predict based on x
        
        loss = loss_func(prediction, target)     # must be (1. nn output, 2. target)
        
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()

        epochLoss += loss

    epochLoss /= 32
    writer.add_scalar('Loss/train', epochLoss, t)


torch.set_printoptions(precision=6)
np.set_printoptions(precision=7, floatmode='fixed', suppress=True)

# prediction = net(normalized_data[0])

# print(x[0].cpu().detach().numpy())
# print()
# print(normalized_data[0].cpu().detach().numpy())
# print()
# print(prediction.cpu().detach().numpy())

# with open('/mnt/c/Users/henri/Documents/Unity Projects/Neural Network/Assets/Motion Matching/NNWeights/LatentVariables.txt', "w+") as f:
#     np.savetxt(f, net.hidden.weight.cpu().detach().numpy().transpose(), delimiter="\n")    
#     np.savetxt(f, net.hidden.bias.cpu().detach().numpy(), delimiter="\n")

#     np.savetxt(f, net.hidden2.weight.cpu().detach().numpy().transpose(), delimiter="\n")    
#     np.savetxt(f, net.hidden2.bias.cpu().detach().numpy(), delimiter="\n")

#     np.savetxt(f, net.hidden3.weight.cpu().detach().numpy().transpose(), delimiter="\n")    
#     np.savetxt(f, net.hidden3.bias.cpu().detach().numpy(), delimiter="\n")

#     np.savetxt(f, net.predict.weight.cpu().detach().numpy().transpose(), delimiter="\n")
#     np.savetxt(f, net.predict.bias.cpu().detach().numpy(), delimiter="\n")