from os import truncate
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch_optimizer as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time

start_time = time.time()

decompressorIn = []
decompressorOut = []
latent = []

with open("/home/pau1o-hs/Documents/Database/XData.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]

        if inner_list == ['']:
            continue

        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        decompressorIn.append(converted)

with open("/home/pau1o-hs/Documents/Database/LatentVariables.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        
        if inner_list == ['']:
            continue
        
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        latent.append(converted)

decompressorIn = np.append(decompressorIn, latent, axis=1)

with open("/home/pau1o-hs/Documents/Database/PoseData.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        
        if inner_list == ['']:
            continue
        
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        decompressorOut.append(converted)

device = torch.device("cuda")

x = torch.tensor(decompressorIn,  dtype=torch.float).to(device)        # x data (tensor), shape=(100, 1)
y = torch.tensor(decompressorOut, dtype=torch.float).to(device)

means = x.mean(dim=1, keepdim=True)
stds = x.std(dim=1, keepdim=True)
normalized_data = (x - means) / stds

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
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=24+32, n_hidden=512, n_output=539).to(device)     # define the network

optimizer = optim.RAdam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

torch.set_printoptions(precision=6)
np.set_printoptions(precision=7, floatmode='fixed', suppress=True)

writer = SummaryWriter()

# train the network
for t in range(10000):
    epochLoss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):

        data.to(device), target.to(device)

        prediction = net(data)     # input x and predict based on x

        loss = loss_func(prediction, target)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()

        epochLoss += loss * prediction.size(0)

    if t % 500 == 0:
        print(t, epochLoss.item())
    
    writer.add_scalar('Python Decompressor Loss', epochLoss, t)

# prediction = net(normalized_data[0])

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

# Export the model
torch.onnx.export(net,                                # model being run
                  normalized_data[0].view(1, 56).float(),         # model dummy input (or a tuple for multiple inputs)
                  "decompressor.onnx",                # where to save the model (can be a file or file-like object)
                  export_params=True,                 # store the trained parameter weights inside the model file
                  opset_version=9,                    # the ONNX version to export the model to
                  do_constant_folding=True,           # whether to execute constant folding for optimization
                  input_names = ['x'],                # the model's input names
                  output_names = ['y']                # the model's output names
                  )

print("Runtime: %s minutes" % ((time.time() - start_time) / 60))