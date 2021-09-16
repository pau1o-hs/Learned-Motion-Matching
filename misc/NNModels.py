from torch.utils.data.dataset import Dataset
import torch
import random
import torch_optimizer as optim
import torch.utils.data as Data
import torch.nn.functional as F

device = torch.device("cuda")

# neural network model
class Compressor(torch.nn.Module):
    # nn layers shape
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_output):
        super(Compressor, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden) 
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.predict = torch.nn.Linear(n_hidden3, n_output)

    # feed forward
    def forward(self, x):
        x = F.elu(self.hidden(x))
        z = F.elu(self.hidden2(x))
        x = F.elu(self.hidden3(z))
        x = self.predict(x)
        return x, z

# neural network model
class Decompressor(torch.nn.Module):
    # nn layers shape
    def __init__(self, n_feature, n_hidden, n_output):
        super(Decompressor, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    # feed forward
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

# neural network model
class Stepper(torch.nn.Module):
    # nn layers shape
    def __init__(self, n_feature, n_hidden, n_output):
        super(Stepper, self).__init__()
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

# neural network model
class Projector(torch.nn.Module):
    # nn layers shape
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(Projector, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2) 
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.hidden4 = torch.nn.Linear(n_hidden3, n_hidden4)
        self.predict = torch.nn.Linear(n_hidden4, n_output)

    # feed forward
    def forward(self, x):
        x = F.relu(self.hidden(x)) 
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.predict(x)
        return x

# override tensor dataset to get a sequence length
class StepperDataset(Dataset):
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

# wip
def compressorLoss(zPred, z):
    lreg = torch.nn.MSELoss()(zPred, z)
    sreg = torch.nn.L1Loss()(zPred, z)
    
    return lreg + sreg

# wip
def decompressorLoss(noiseInput, predict, target):
    l2loss = torch.nn.L1Loss()(torch.nn.MSELoss()(predict[:, :24], noiseInput), torch.nn.MSELoss()(target[:, :24], noiseInput))
    loss = torch.nn.L1Loss()(predict, target) + l2loss
    return loss

# wip
def stepperLoss(data, predict, target):
    l1loss = torch.nn.L1Loss(target - (data + predict)).item()
    loss = l1loss + torch.nn.L1Loss(((data[:19] - target[1:]) / 0.017) - ((predict[:19] - predict[1:]) / 0.017)).item()
    return loss

# wip
def projectorLoss(noiseInput, predict, target):
    l2loss = torch.nn.L1Loss()(torch.nn.MSELoss()(predict[:, :24], noiseInput), torch.nn.MSELoss()(target[:, :24], noiseInput))
    loss = torch.nn.L1Loss()(predict, target) + l2loss
    return loss

# dataloader settings for training
def TrainSettings(model):
    optimizer = optim.RAdam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

    return optimizer, scheduler