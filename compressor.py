import sys
sys.path.append('./misc')

from torch.utils.tensorboard import SummaryWriter
import time
import torch
import NNModels
import CustomFunctions
import numpy as np
import torch_optimizer as optim
import torch.utils.data as Data
import torch.nn.functional as F

# runtime start
start_time = time.time()

# checking if GPU is available
print(torch.cuda.get_device_name(0)) 
device = torch.device("cuda")

# define nn params
model = NNModels.Compressor(n_feature=1232, n_hidden=512, n_hidden2=32, n_hidden3=512, n_output=1232).to(device)

# load data
compressorIn = CustomFunctions.LoadData("YData")

# convert list to tensor
x = torch.tensor(compressorIn, dtype=torch.float).to(device)
y = torch.tensor(compressorIn, dtype=torch.float).to(device)

# normalize data
normX = CustomFunctions.NormalizeData(x)

# dataloader settings for training
train = Data.TensorDataset(normX, normX)
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
        prediction = model(data, False)
        
        # MSELoss: prediction, target
        loss = torch.nn.MSELoss()(prediction, target)     
        
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()        # step learning rate schedule
        
        # log loss value
        epochLoss += loss * prediction.size(0)

    # just printing where training is
    if t % 500 == 0:
        print(t, epochLoss.item())
    
    # log loss in tensorboard
    writer.add_scalar('Python Compressor Loss', epochLoss, t)

# saving latent variables (z)
with open('/home/pau1o-hs/Documents/Database/LatentVariables.txt', "w+") as f:
    for i in range(len(normX)):
        prediction = model(normX[i], True)
        np.savetxt(f, prediction.cpu().detach().numpy()[None], delimiter=" ")

# export the model
torch.onnx.export(
    model, torch.rand(1, 1232, device=device), 
    "onnx/compressor.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Compressor Runtime: %s minutes" % ((time.time() - start_time) / 60))