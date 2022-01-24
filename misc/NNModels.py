from torch.utils.data.dataset import Dataset
import torch
import random
import numpy as np
import torch_optimizer as optim
import torch.utils.data as Data
import torch.nn.functional as F

device = torch.device("cuda")

# neural network model
class Compressor(torch.nn.Module):
    # nn layers shape
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Compressor, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size) 
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.predict = torch.nn.Linear(hidden_size, output_size)

    # feed forward
    def forward(self, x):
        x = F.elu(self.layer1(x))
        x = F.elu(self.layer2(x))
        x = F.elu(self.layer3(x))
        x = self.predict(x)
        return x

# neural network model
class Decompressor(torch.nn.Module):
    # nn layers shape
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Decompressor, self).__init__()
        self.layer1  = torch.nn.Linear(input_size, hidden_size)
        self.predict = torch.nn.Linear(hidden_size, output_size)
    
    # feed forward
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.predict(x)
        return x

# neural network model
class Stepper(torch.nn.Module):
    # nn layers shape
    def __init__(self, input_size, hidden_size=512):
        super(Stepper, self).__init__()
        self.layer1  = torch.nn.Linear(input_size, hidden_size)
        self.layer2  = torch.nn.Linear(hidden_size, hidden_size)
        self.predict = torch.nn.Linear(hidden_size, input_size)

    # feed forward
    def forward(self, x):
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x))
        x = self.predict(x)
        return x

# neural network model
class Projector(torch.nn.Module):
    # nn layers shape
    def __init__(self, input_size, output_size, hidden_size=512):
        super(Projector, self).__init__()
        self.layer1  = torch.nn.Linear(input_size, hidden_size)  
        self.layer2  = torch.nn.Linear(hidden_size, hidden_size) 
        self.layer3  = torch.nn.Linear(hidden_size, hidden_size)
        self.layer4  = torch.nn.Linear(hidden_size, hidden_size)
        self.predict = torch.nn.Linear(hidden_size, output_size)

    # feed forward
    def forward(self, x):
        x = F.relu(self.layer1(x)) 
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.predict(x)
        return x

# override tensor dataset to get a sequence length
class CustomDataset(Dataset):
    def __init__(self, datas, indices, window=1):
        self.datas = datas
        self.window = window
        self.samples = []

        # Build batches respecting window size
        for i in range(len(indices)):
            for j in range(indices[i] - window):
                self.samples.append(np.arange(j, j + window))

    def __getitem__(self, index):
        output = ()
        index = self.samples[np.random.randint(0, len(self.samples))]
        
        for i in range(len(self.datas)):
            output = output + (self.datas[i][index],)

        return output

    def __len__(self):
        return self.datas[0].size(0)

# dataloader settings for training
def TrainSettings(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    return optimizer, scheduler