import sys
import os.path
import os, psutil
sys.path.append('./misc')

from torch.utils.tensorboard import SummaryWriter
import time
import torch
import NNModels
import CustomFunctions
import numpy as np
import torch_optimizer as optim

# runtime start
start_time = time.time()

# checking if GPU is available
print(torch.cuda.get_device_name(0)) 
device = torch.device("cuda")

# define nn params
compressor   = NNModels.Compressor(n_feature=2002, n_hidden=512, n_hidden2=512, n_hidden3=512, n_output=32).to(device)
decompressor = NNModels.Decompressor(n_feature=24+32, n_hidden=512, n_output=1001).to(device)

# load data
xData = CustomFunctions.LoadData("XData", True)
yData = CustomFunctions.LoadData("YData", True)

hierarchy = CustomFunctions.LoadData("HierarchyData")
hierarchy = [int(i) for i in hierarchy[0]]

# list of tensors
xTensor = []
yTensor = []
qTensor = []

for i in range(len(xData)):
    xTensor.append(torch.tensor(xData[i], dtype=torch.float).to(device))
    yTensor.append(torch.tensor(yData[i], dtype=torch.float).to(device))

# compute forward kinematics
qTensor = CustomFunctions.ForwardKinematics(yTensor, hierarchy)
# print()

# relevant dataset indexes
transforms = list(range(0, 13))
velocities = []

[transforms.extend(list(range(i, i + 7))) for i in range(13, yTensor[0].size(1), 13)]
[velocities.extend(list(range(i + 7, i + 13))) for i in range(0, yTensor[0].size(1), 13)]

# dataloader settings
train = NNModels.CustomDataset((xTensor, yTensor, qTensor), window=2)
train_loader = NNModels.Data.DataLoader(train, shuffle=True, batch_size=32)
c_optimizer, c_scheduler = NNModels.TrainSettings(compressor)
d_optimizer, d_scheduler = NNModels.TrainSettings(decompressor)

# training settings
epochs = 15000
logFreq = 500

# init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(epochs + 1):
    epochLoss = 0.0
    epoch_time = time.time()

    for batch_idx, (x, y, q) in enumerate(train_loader):

        x.to(device), y.to(device)
        
        # (seq. length, batch size, features) <- (batch size, seq. length, features)
        x = torch.transpose(x, 0, 1)
        y = torch.transpose(y, 0, 1)
        q = torch.transpose(q, 0, 1)

        # generate latent variables Z
        p = CustomFunctions.NormalizeData(torch.cat((y, q), dim=2), dim=2)
        zPred = compressor(p)
        
        # reconstruct pose Y
        fData = CustomFunctions.NormalizeData(torch.cat((x, zPred), dim=2), dim=2)
        yPred = decompressor(fData)

        # recompute forward kinematics
        # qPred = torch.stack(CustomFunctions.ForwardKinematics(yPred, hierarchy), dim=0)

        #  compute latent regularization losses
        loss_lreg = torch.nn.MSELoss()(zPred, torch.zeros_like(zPred))
        # loss_sreg = torch.nn.L1Loss() (zPred, torch.zeros_like(zPred))
        # loss_vreg = torch.nn.MSELoss()((zPred[1] - zPred[0]) / 0.017, torch.zeros_like(zPred[0]))

        # local & character space losses
        loss_loc = torch.nn.L1Loss()(yPred[:,:,transforms], y[:,:,transforms])
        # loss_chr = torch.nn.L1Loss()(qPred[:,:,transforms], q[:,:,transforms])

        # local & character space velocity losses
        loss_lvel = torch.nn.L1Loss()((yPred[1,:,transforms] - yPred[0,:,transforms]) / 0.017, (y[1,:,transforms] - y[0,:,transforms]) / 0.017)
        # loss_cvel = torch.nn.L1Loss()((qPred[1,:,transforms] - qPred[0,:,transforms]) / 0.017, (q[1,:,transforms] - q[0,:,transforms]) / 0.017)

        # loss = loss_lreg + loss_sreg + loss_vreg + loss_loc + loss_chr + loss_lvel + loss_cvel
        loss = loss_lreg + loss_loc + loss_lvel

        # clear gradients for next train
        c_optimizer.zero_grad()
        d_optimizer.zero_grad()

        # backpropagation, compute gradients
        loss.backward()

        # apply gradients
        c_optimizer.step()
        d_optimizer.step()

        # step learning rate schedule
        c_scheduler.step()
        d_scheduler.step()

        # log loss value
        epochLoss += loss.item()
        
    # just printing where training is
    if t % logFreq == 0:
        print("Epoch:", t, "\tLoss:", epochLoss, "\tRuntime:", (time.time() - epoch_time) * logFreq / 60, "minutes")
    
    # log loss in tensorboard
    writer.add_scalar('Decompressor Loss', epochLoss, t)

# save character space transforms (Q)
with open('database/QData.txt', "w+") as f:
    for i in range(len(qTensor)):        
        for j in range(qTensor[i].size(0)):
            np.savetxt(f, qTensor[i][j].cpu().detach().numpy()[None], delimiter=" ")

        np.savetxt(f, [''], fmt='%s')

# save latent variables (Z)
with open('database/ZData.txt', "w+") as f:
    for i in range(len(yTensor)):
        p = CustomFunctions.NormalizeData(torch.cat((yTensor[i], qTensor[i]), dim=1), dim=1)

        for j in range(p.size(0)):
            zPred = compressor(p[i])
            np.savetxt(f, zPred.cpu().detach().numpy()[None], delimiter=" ")

        np.savetxt(f, [''], fmt='%s')

# export compressor model
torch.onnx.export(
    compressor, torch.rand(1, 1, 2002, device=device), 
    "onnx/compressor.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['z']
)

# export decompressor model
torch.onnx.export(
    decompressor, torch.rand(1, 1, 56, device=device), 
    "onnx/decompressor.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Decompressor runtime: %s minutes" % ((time.time() - start_time) / 60))