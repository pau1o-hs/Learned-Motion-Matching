from torch.utils.tensorboard import SummaryWriter
import time
import torch
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
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    
    # feed forward
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

# define nn params
model = Model(n_feature=24+32, n_hidden=512, n_output=539).to(device)

# load data
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

with open("/home/pau1o-hs/Documents/Database/PoseData.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        
        if inner_list == ['']:
            continue
        
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        decompressorOut.append(converted)

# combined features (x + z)
decompressorIn = np.append(decompressorIn, latent, axis=1)

# convert list to tensor
x = torch.tensor(decompressorIn,  dtype=torch.float).to(device)        # x data (tensor), shape=(100, 1)
y = torch.tensor(decompressorOut, dtype=torch.float).to(device)

# normalize data
means = x.mean(dim=1, keepdim=True)
stds = x.std(dim=1, keepdim=True)
normalized_data = (x - means) / stds

# dataloader settings for training
train = Data.TensorDataset(normalized_data, y)
train_loader = Data.DataLoader(train, shuffle=True, batch_size=32)

optimizer = optim.RAdam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

# init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(1):
    epochLoss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):

        data.to(device), target.to(device)

        # feed forward
        prediction = model(data)

        # MSELoss: prediction, target
        loss = torch.nn.MSELoss()(prediction, target)   

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
    writer.add_scalar('Python Decompressor Loss', epochLoss, t)

# export the model
torch.onnx.export(
    model, torch.rand(1, 56, device=device),
    "decompressor.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Decompressor Runtime: %s minutes" % ((time.time() - start_time) / 60))