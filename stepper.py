from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
import time
import torch
import random
import numpy as np
import torch_optimizer as optim
import torch.utils.data as Data
import torch.nn.functional as F

# runtime start
start_time = time.time()

# checking if GPU is available
print(torch.cuda.get_device_name(0)) 
device = torch.device("cuda")

# neural network model
class Model(torch.nn.Module):
    # nn layers shape
    def __init__(self, n_feature, n_hidden, n_output):
        super(Model, self).__init__()
        self.n_hidden = n_hidden

        self.hidden  = torch.nn.LSTM(n_feature, n_hidden, 1)
        self.hidden2  = torch.nn.LSTM(n_hidden, n_hidden, 1)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # feed forward
    def forward(self, x, h_t, c_t): 
        h_t2 = torch.zeros(1, x.size(1), 512, device=device).requires_grad_()
        c_t2 = torch.zeros(1, x.size(1), 512, device=device).requires_grad_() 

        output, (h_t, c_t) = self.hidden(x, (h_t, c_t))
        output = F.relu(output)

        output, (h_t2, c_t2) = self.hidden2(output, (h_t2, c_t2))
        output = F.relu(output)

        output = self.predict(output)
        return output, h_t2, c_t2

# loss function (wip)
def customLoss(data, predict, target):
    l1loss = torch.nn.L1Loss(target - (data + predict)).item()
    loss = l1loss + torch.nn.L1Loss(((data[:19] - target[1:]) / 0.017) - ((predict[:19] - predict[1:]) / 0.017)).item()
    return loss

# define nn params
net = Model(n_feature=24+32, n_hidden=512, n_output=24+32).to(device)

# load data
clip = []
stepperInput = []
stepperNext = []
latent = []

with open("/home/pau1o-hs/Documents/Database/XData.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]

        if inner_list == ['']:
            stepperInput.append(clip)
            clip = []
            continue

        converted = []
        for item in inner_list:
            converted.append(float(item))
        
        clip.append(converted)

with open("/home/pau1o-hs/Documents/Database/LatentVariables.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        
        if inner_list == ['']:
            continue
        
        converted = []
        for item in inner_list:
            converted.append(float(item))
        latent.append(converted)

# combined features (x + z)
latentCount = 0
for i in range(len(stepperInput)):
    for j in range(len(stepperInput[i])):
        stepperInput[i][j].extend(latent[latentCount])
        latentCount += 1

# output: input shifted -1
stepperNext = stepperInput
for i in range(len(stepperInput)):
    for j in range(len(stepperInput[i])):
        stepperNext[i][j].append(stepperNext[i][j].pop(0))

# normalize data
x = []
y = []
normalized_input = []
normalized_output = []

for i in range(len(stepperInput)):
    # list of tensors (each tensor represent a clip)
    x.append(torch.tensor(stepperInput[i], device=device))
    y.append(torch.tensor(stepperNext[i], device=device))

    means = x[i].mean(dim=1, keepdim=True)
    stds = x[i].std(dim=1, keepdim=True)
    normalized_input.append((x[i] - means) / stds)

    means = y[i].mean(dim=1, keepdim=True)
    stds = y[i].std(dim=1, keepdim=True)
    normalized_output.append((y[i] - means) / stds)

# override tensor dataset to get a sequence length
class CustomDataset(Dataset):
    def __init__(self, dataX, dataY, window=20):
        self.dataX = dataX
        self.dataY = dataY
        self.window = window
        self.selectedClip = 0

    def __getitem__(self, index):
        frameIndex = random.randint(0, len(self.dataX[self.selectedClip]) - self.window - 1)
        x = self.dataX[self.selectedClip][frameIndex:frameIndex+self.window]
        y = self.dataY[self.selectedClip][frameIndex:frameIndex+self.window]
        return x, y

    def __len__(self):
        self.selectedClip = random.randint(0, len(self.dataX) - 1)
        return len(self.dataX[self.selectedClip])

# dataloader settings for training
dataSet = CustomDataset(normalized_input, normalized_output)
train_loader = Data.DataLoader(dataSet, shuffle=True, batch_size=32)

optimizer = optim.RAdam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

# init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(20001):
    epochLoss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):    
        
        data.to(device), target.to(device)

        # data: (batch size, seq. length)
        # LSTM: (seq. length, batch size, hidden nodes)
        data = torch.transpose(data, 0, 1)
        target = torch.transpose(target, 0, 1)
        n_batch = data.size(1)

        h_t = torch.zeros(1, n_batch, 512, device=device).requires_grad_()
        c_t = torch.zeros(1, n_batch, 512, device=device).requires_grad_()

        # feed forward
        prediction, h_t, c_t = net(data, h_t, c_t)

        # MSELoss: prediction, target
        loss = torch.nn.L1Loss()(prediction, target)   

        # clear gradients for next train
        optimizer.zero_grad()

        # backpropagation, compute gradients
        loss.backward()

        # apply gradients
        optimizer.step()

        # step learning rate schedule
        scheduler.step()

        # log loss value
        epochLoss += loss * prediction.size(0)

    # just printing where training is
    if t % 500 == 0:
        print(t, epochLoss.item())
    
    # log loss in tensorboard    
    writer.add_scalar('Python Stepper Loss', epochLoss, t)

# sample input shape
x = torch.rand(1, 1, 56, device=device)
h_t = torch.rand(1, 1, 512, device=device)
c_t = torch.rand(1, 1, 512, device=device)

# export the model
torch.onnx.export(
    net, (x, h_t, c_t),
    "stepper.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x', 'h0', 'c0'], output_names =['y', 'hn', 'cn']
)

# runtime end
print("Stepper Runtime: %s minutes" % ((time.time() - start_time) / 60))