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
model = NNModels.Decompressor(n_feature=24+32, n_hidden=512, n_output=539).to(device)

# load data
decompressorIn = CustomFunctions.LoadData("XData")
decompressorOut = CustomFunctions.LoadData("PoseData")
latent = CustomFunctions.LoadData("LatentVariables")

# combined features (x + z)
decompressorIn = np.append(decompressorIn, latent, axis=1)

# convert list to tensor
x = torch.tensor(decompressorIn,  dtype=torch.float).to(device)        # x data (tensor), shape=(100, 1)
y = torch.tensor(decompressorOut, dtype=torch.float).to(device)

# normalize data
normX = CustomFunctions.NormalizeData(x)

# dataloader settings for training
train = Data.TensorDataset(normX, y)
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
    writer.add_scalar('Python Decompressor Loss', epochLoss, t)

# export the model
torch.onnx.export(
    model, torch.rand(1, 56, device=device),
    "onnx/decompressor.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Decompressor Runtime: %s minutes" % ((time.time() - start_time) / 60))