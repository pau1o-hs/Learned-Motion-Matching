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
compressor   = NNModels.Compressor(n_feature=1001, n_hidden=512, n_hidden2=32, n_hidden3=512, n_output=1001).to(device)
decompressor = NNModels.Decompressor(n_feature=24+32, n_hidden=512, n_output=1001).to(device)

# load data
xData = CustomFunctions.LoadData("XData", True)
yData = CustomFunctions.LoadData("YData", True)

hierarchy = CustomFunctions.LoadData("HierarchyData")
hierarchy = [int(i) for i in hierarchy[0]]

xTensor = []
yTensor = []
qTensor = []
zTensor = []

# list of tensors
for i in range(len(xData)):
    xTensor.append(torch.tensor(xData[i], dtype=torch.float).to(device))
    yTensor.append(torch.tensor(yData[i], dtype=torch.float).to(device))

qTensor = CustomFunctions.ForwardKinematics(yTensor, hierarchy)

for i in range(len(qTensor)):
    zTensor.append(torch.cat((yTensor[i], qTensor[i]), dim=1))
    zTensor[i] = CustomFunctions.NormalizeData(zTensor[i])

print()

# relevant dataset indexes
localTransforms = []
globalTransforms = []
velocities = []
rootVelocity = range(7, 13)

[localTransforms.extend(list(range(i, i + 7))) for i in range(0, yTensor[0].size(1), 13)]
[globalTransforms.extend(list(range(i, i + 7))) for i in range(1001, zTensor[0].size(1), 13)]
[velocities.extend(list(range(i + 7, i + 13))) for i in range(0, zTensor[0].size(1), 13)]

# dataloader settings
train = NNModels.StepperDataset((xTensor, yTensor, qTensor, zTensor), window=2)
train_loader = NNModels.Data.DataLoader(train, shuffle=True, batch_size=32)
c_optimizer, c_scheduler = NNModels.TrainSettings(compressor)
d_optimizer, d_scheduler = NNModels.TrainSettings(decompressor)

# c_optimizer = optim.RAdam(compressor.parameters(), lr=0.001, weight_decay=1e-3)
# c_scheduler = torch.optim.lr_scheduler.StepLR(c_optimizer, step_size=1000, gamma=0.99)

# training settings
epochs = 10001
logFreq = 500

# init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(epochs + 1):
    epochLoss = 0.0
    epoch_time = time.time()

    for batch_idx, (x, y, q, z) in enumerate(train_loader):

        x.to(device), y.to(device), q.to(device), z.to(device)
        
        # (seq. length, batch size, features) <- (batch size, seq. length, features)
        x = torch.transpose(x, 0, 1)
        y = torch.transpose(y, 0, 1)
        q = torch.transpose(q, 0, 1)
        z = torch.transpose(z, 0, 1)

        # generate latent variables z_
        zPred, z_ = compressor(z[:,:,:1001])

        # reconstruct pose ỹ
        combinedFeatures = torch.cat((x, z_), dim=2)
        combinedFeatures = CustomFunctions.NormalizeData(combinedFeatures, dim=2)
        yPred = decompressor(combinedFeatures)

        # recompute forward kinematics
        # qPred = CustomFunctions.ForwardKinematics(yPred, hierarchy)
        # qPred = torch.cat((qPred[0].unsqueeze(0), qPred[1].unsqueeze(0)), dim=0)

        #  compute latent regularization losses
        # loss_lreg = torch.nn.MSELoss()(zPred[:,localTransforms], torch.zeros_like(zPred[:,localTransforms]))
        # loss_sreg = torch.nn.L1Loss()(zPred[:,globalTransforms], torch.zeros_like(zPred[:,globalTransforms]))

        # local & character space losses
        loss_loc  = torch.nn.L1Loss()(yPred[:,:,localTransforms], y[:,:,localTransforms])
        # loss_chr  = torch.nn.L1Loss()(qPred[:,:,localTransforms], q[:,:,localTransforms])

        # local & character space velocity losses
        loss_lvel = torch.nn.L1Loss()((yPred[0,:,rootVelocity] - yPred[1,:,rootVelocity]) / 0.017, (y[0,:,rootVelocity] - y[1,:,rootVelocity]) / 0.017)
        # loss_cvel = torch.nn.L1Loss()((qPred[0,:,rootVelocity] - qPred[1,:,rootVelocity]) / 0.017, (q[0,:,rootVelocity] - q[1,:,rootVelocity]) / 0.017)

        # losses sum
        # loss = loss_loc + loss_chr + loss_lvel + loss_cvel
        loss = loss_loc + loss_lvel

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

        # print('Batch', batch_idx, 'done.')
        
    # just printing where training is
    if t % logFreq == 0:
        print("Epoch:", t, "\tLoss:", epochLoss, "\tRuntime:", (time.time() - epoch_time) * logFreq / 60, "minutes")
    
    # log loss in tensorboard
    writer.add_scalar('Python Decompressor Loss', epochLoss, t)

# saving latent variables (z_)
with open('database/zData.txt', "w+") as f:
    for i in range(len(zTensor)):
        for j in range(len(zTensor[i])):
            zPred, z_ = compressor(zTensor[i][j])
            np.savetxt(f, z_.cpu().detach().numpy()[None], delimiter=" ")

# export compressor model
torch.onnx.export(
    compressor, torch.rand(1, 2002, device=device), 
    "onnx/compressor.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['Z', 'z']
)

# export decompressor model
torch.onnx.export(
    decompressor, torch.rand(1, 56, device=device), 
    "onnx/decompressor.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Compressor runtime: %s minutes" % ((time.time() - start_time) / 60))