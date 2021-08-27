from numpy.random import random_integers
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
import torch_optimizer as optim
import random
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time

start_time = time.time()

stepperInput = []
stepperNext = []
latent = []

with open("/home/pau1o-hs/Documents/Database/XData.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]

        if inner_list == ['']:
            continue

        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        stepperInput.append(converted)

with open("/home/pau1o-hs/Documents/Database/LatentVariables.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        
        if inner_list == ['']:
            continue
        
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        latent.append(converted)

device = torch.device("cuda")

stepperInput = np.append(stepperInput, latent, axis=1)
stepperNext = np.roll(stepperInput, -1, axis=0)

x = torch.tensor(stepperInput, dtype=torch.float).to(device)        # x data (tensor), shape=(100, 1)
y = torch.tensor(stepperNext,  dtype=torch.float).to(device)

means = x.mean(dim=1, keepdim=True)
stds = x.std(dim=1, keepdim=True)
normalized_input = (x - means) / stds

means = y.mean(dim=1, keepdim=True)
stds = y.std(dim=1, keepdim=True)
normalized_output = (y - means) / stds

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

train = Data.TensorDataset(normalized_input, normalized_output)

dataSet = MyDataset(train)

train_loader = Data.DataLoader(dataSet, shuffle=True, batch_size=32)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_output):
        super(Net, self).__init__()
        self.n_hidden = n_hidden
        self.n_hidden2 = n_hidden2
        self.n_output = n_output

        self.hidden  = torch.nn.LSTM(n_feature, n_hidden, 1, batch_first=True)   # hidden layer
        self.hidden2 = torch.nn.LSTM(n_hidden, n_hidden2, 1, batch_first=True)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

        self.gru = torch.nn.GRU(n_hidden, n_hidden)

    def forward(self, x, future=10):
        outputs = []
        n_batch = x.size(0)
        n_samples = x.size(1)

        h_t = torch.zeros(1, n_batch, self.n_hidden, dtype=torch.float32, device=device)
        c_t = torch.zeros(1, n_batch, self.n_hidden, dtype=torch.float32, device=device)
        h_t2 = torch.zeros(1, n_batch, self.n_hidden, dtype=torch.float32, device=device)
        c_t2 = torch.zeros(1, n_batch, self.n_hidden, dtype=torch.float32, device=device)
        
        h_t, c_t = self.hidden(x, (h_t, c_t))
        h_t = F.relu(h_t)

        h_t2, c_t2 = self.hidden2(h_t, (h_t2, c_t2))
        h_t2 = F.relu(h_t2)

        output = self.predict(h_t2)
        outputs.append(output)

        # output = torch.cat(output, dim=0).split(1, dim=1).split(1, dim=2)
        # print(output.shape)

        # for i in range(future):
        #     print(output.split(1, dim=0)[-1].shape)
        #     h_t, c_t = F.relu(self.hidden(output.split(1, dim=0)[-1], (h_t, c_t)))
        #     h_t2, c_t2 = F.relu(self.hidden(h_t, (h_t2, c_t2)))
        #     output = self.predict(h_t2)
        #     outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        return output

        # x = F.relu(self.hidden(x))      # activation function for hidden layer
        # x = F.relu(self.hidden2(x))     # activation function for hidden layer
        # x = self.predict(x)             # linear output
        # return x

net = Net(n_feature=24+32, n_hidden=512, n_hidden2=512, n_output=24+32).to(device)     # define the network

# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
optimizer = optim.RAdam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

writer = SummaryWriter()

# train the network
for t in range(1):
    epochLoss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):    
        
        data.to(device), target.to(device) 

        prediction = net(data)     # input x and predict based on x
        # print(prediction.shape)

        delta = target - data
        loss = loss_func(prediction, delta)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()

        epochLoss += loss * prediction.size(0)
        # print("batchComplete")

    if t % 500 == 0:
        print(t, epochLoss.item())
    
    writer.add_scalar('Python Stepper Loss', epochLoss, t)

torch.set_printoptions(precision=6)
np.set_printoptions(precision=7, floatmode='fixed', suppress=True)

# ar_hidden = net.initHidden(20)
# prediction, ar_hidden = net(normalized_input[0], ar_hidden)

# print(x[0].detach().numpy())
# print()
# print(y[0].detach().numpy())
# print()
# print(prediction.detach().numpy())

with torch.no_grad():
    inputx = torch.randn(1, 20, 56).to(device)

    # Export the model
    torch.onnx.export(net,                              # model being run
                    inputx,                  # model dummy input (or a tuple for multiple inputs)
                    "stepper.onnx",                     # where to save the model (can be a file or file-like object)
                    export_params=True,                 # store the trained parameter weights inside the model file
                    opset_version=9,                    # the ONNX version to export the model to
                    do_constant_folding=True,           # whether to execute constant folding for optimization
                    input_names = ['x', 'h0', 'c0'],    # the model's input names
                    output_names = ['y'],               # the model's output names
                    dynamic_axes={'x': {0: 'sequence'}, 'y': {0: 'sequence'}}
                  )

# with open('/home/pau1o-hs/Documents/NNWeights/Stepper.txt', "w+") as f:
#     np.savetxt(f, net.hidden.weight.cpu().detach().numpy().transpose(), delimiter="\n")
#     np.savetxt(f, net.hidden.bias.cpu().detach().numpy(), delimiter="\n")

#     np.savetxt(f, net.hidden2.weight.cpu().detach().numpy().transpose(), delimiter="\n")
#     np.savetxt(f, net.hidden2.bias.cpu().detach().numpy(), delimiter="\n")

#     np.savetxt(f, net.predict.weight.cpu().detach().numpy().transpose(), delimiter="\n")
#     np.savetxt(f, net.predict.bias.cpu().detach().numpy(), delimiter="\n")

    # np.savetxt(f, np.append(net.gru.all_weights[0][0].cpu().detach().numpy(), net.gru.all_weights[0][2].view(1536, 1).cpu().detach().numpy(), axis=1).transpose(), delimiter="\n")    
    # np.savetxt(f, np.append(net.gru.all_weights[0][1].cpu().detach().numpy(), net.gru.all_weights[0][3].view(1536, 1).cpu().detach().numpy(), axis=1).transpose(), delimiter="\n")

# nWeights = torch.numel(net.hidden.weight)
# nWeights += torch.numel(net.hidden.bias)

# nWeights += torch.numel(net.hidden2.weight)
# nWeights += torch.numel(net.hidden2.bias)

# nWeights += torch.numel(net.predict.weight)
# nWeights += torch.numel(net.predict.bias)

# nWeights += torch.numel(net.gru.all_weights[0][0])
# nWeights += torch.numel(net.gru.all_weights[0][1])
# nWeights += torch.numel(net.gru.all_weights[0][2])
# nWeights += torch.numel(net.gru.all_weights[0][3])

# print(net.hidden.weight.shape)
# print(net.hidden.bias.shape)
# print(net.hidden2.weight.shape)
# print(net.hidden2.bias.shape)
# print(net.predict.weight.shape)
# print(net.predict.bias.shape)
# print(nWeights)
# print(net.gru.all_weights[0][0][512])
# print(net.gru.all_weights[0][0].size(0), net.gru.all_weights[0][0].size(1))
# print(net.gru.all_weights[0][1].size(0), net.gru.all_weights[0][1].size(1))
# print(net.gru.all_weights[0][2].size(0), net.gru.all_weights[0][2].size(0))
# print(net.gru.all_weights[0][3].size(0), net.gru.all_weights[0][3].size(0))

print("Runtime: %s minutes" % ((time.time() - start_time) / 60))