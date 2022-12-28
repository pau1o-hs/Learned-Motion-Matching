import sys
sys.path.append('./misc')

from torch.utils.tensorboard import SummaryWriter
import time
import torch
import NNModels
import CustomFunctions
import numpy as np

# Runtime start
start_time = time.time()

# Checking if GPU is available
print(torch.cuda.get_device_name(0)) 
device = torch.device("cpu")

# Load data
X = CustomFunctions.LoadData("XData")['data']
Z = CustomFunctions.LoadData("ZData")['data']
indices = CustomFunctions.LoadData("XData")['indices']

# To tensor
X = torch.as_tensor(X, dtype=torch.float).to(device)
Z = torch.as_tensor(Z, dtype=torch.float).to(device)

dt = 1.0 / 60.0

# Scales
X_scale = X.std()
Z_scale = Z.std()

# Means and stds
stepper_mean_in = torch.cat((X, Z), dim=-1).mean(dim=0)
stepper_std_in  = torch.cat((X_scale.repeat(24), Z_scale.repeat(32)), dim=0)

Xvel = (X[indices[0]+1:indices[1]] - X[indices[0]:indices[1]-1]) / dt

for i in range (1, len(indices) - 1):
    Xvel = torch.cat((Xvel, (X[indices[i]+1:indices[i + 1]] - X[indices[i]:indices[i + 1]-1]) / dt), dim=0)

Zvel = (Z[indices[0]+1:indices[1]] - Z[indices[0]:indices[1]-1]) / dt

for i in range (1, len(indices) - 1):
    Zvel = torch.cat((Zvel, (Z[indices[i]+1:indices[i + 1]] - Z[indices[i]:indices[i + 1]-1]) / dt), dim=0)

stepper_mean_out = torch.cat((Xvel, Zvel), dim=-1).mean(dim=0)
stepper_std_out  = torch.cat((Xvel, Zvel), dim=-1).std(dim=0) + 0.001

# Training settings
nfeatures = X.size(1)
nlatent = 32
epochs = 10000
batchsize=32
window=20
logFreq = 500
dt = 1.0 / 60.0

stepper = NNModels.Stepper(nfeatures + nlatent).to(device)

# dataloader settings for training
optimizer, scheduler = NNModels.TrainSettings(stepper)

# Build batches respecting window size
samples = []
for i in range(len(indices)):
    for j in range(indices[i] - window):
        samples.append(np.arange(j, j + window))
samples = torch.as_tensor(np.array(samples))

# init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(epochs + 1):
    epoch_time = time.time()

    # batch
    batch = samples[torch.randint(0, len(samples), size=[batchsize])]

    Xgnd = X[batch.long()]
    Zgnd = Z[batch.long()]

    # Predict delta x and delta z over a window of s frames
    Xtil = [Xgnd[:,0]]
    Ztil = [Zgnd[:,0]]

    for i in range(1, window):
        pred = (stepper((torch.cat((Xtil[i - 1], Ztil[i - 1]), dim=-1) - 
            stepper_mean_in) / stepper_std_in) *
            stepper_std_out + stepper_mean_out)

        Xtil.append(Xtil[-1] + pred[:,:nfeatures] * dt)
        Ztil.append(Ztil[-1] + pred[:,nfeatures:] * dt)
        
    Xtil = torch.cat([x[:,None] for x in Xtil], dim=1)
    Ztil = torch.cat([z[:,None] for z in Ztil], dim=1)
    
    # Compute velocities
    Xgnd_vel = (Xgnd[:,1:] - Xgnd[:,:-1]) / dt
    Zgnd_vel = (Zgnd[:,1:] - Zgnd[:,:-1]) / dt
    
    Xtil_vel = (Xtil[:,1:] - Xtil[:,:-1]) / dt
    Ztil_vel = (Ztil[:,1:] - Ztil[:,:-1]) / dt

    # Compute losses
    loss_xval = torch.mean(30. * torch.abs(Xgnd - Xtil))
    loss_zval = torch.mean(7.0 * torch.abs(Zgnd - Ztil))
    loss_xvel = torch.mean(1.2 * torch.abs(Xgnd_vel - Xtil_vel))
    loss_zvel = torch.mean(0.1 * torch.abs(Zgnd_vel - Ztil_vel))

    loss = loss_xval + loss_zval + loss_xvel + loss_zvel

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    # Step learning rate schedule
    if t % 1000 == 0:
        scheduler.step()

    # Log
    writer.add_scalar('stepper/loss', loss.item(), t)

    writer.add_scalars('stepper/loss_terms', {
        'xval': loss_xval.item(),
        'zval': loss_zval.item(),
        'xvel': loss_xvel.item(),
        'zvel': loss_zvel.item(),
    }, t)

    if t % logFreq == 0:
        print("Epoch:", t, "\tLoss:", loss.item(), "\tRuntime:", (time.time() - epoch_time) * logFreq / 60, "minutes")
    

# sample input shape
x = torch.rand(1, 1, 56, device=device)

# export the model
torch.onnx.export(
    stepper, x,
    "onnx/stepper.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names =['y']
)

# runtime end
print("Stepper runtime: %s minutes" % ((time.time() - start_time) / 60))