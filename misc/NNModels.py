from torch.utils.data.dataset import Dataset
import torch
import random
import torch_optimizer as optim
import torch.utils.data as Data
import torch.nn.functional as F
from copy import deepcopy

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
    def __init__(self, datas, window=1):
        self.datas = datas
        self.window = window
        self.selectedClip = 0
        self.samples = []
        self.fullBatch = []

        # clips length
        for i in range(len(self.datas[0])):
            # frames length
            for j in range(self.datas[0][i].size(0) - self.window + 1):
                self.samples.append((i, j))
            
    def __getitem__(self, index):
        output = ()
        index = self.fullBatch.pop()

        for i in range(len(self.datas)):
            output = output + (self.datas[i][index[0]][index[1]:index[1]+self.window],)

        return output

    def __len__(self):
        self.fullBatch = self.samples.copy()

        random.shuffle(self.fullBatch)
        return len(self.fullBatch)

# dataloader settings for training
def TrainSettings(model):
    optimizer = optim.RAdam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

    return optimizer, scheduler