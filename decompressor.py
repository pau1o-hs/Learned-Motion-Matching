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

# runtime start
start_time = time.time()

# checking if GPU is available
print(torch.cuda.get_device_name(0)) 
device = torch.device("cuda")

# define nn params
compressor   = NNModels.Compressor(n_feature=2002, n_hidden=512, n_hidden2=32, n_hidden3=512, n_output=2002).to(device)
decompressor = NNModels.Decompressor(n_feature=24+32, n_hidden=512, n_output=1001).to(device)

# load data
xData = CustomFunctions.LoadData("XData")
yData = CustomFunctions.LoadData("YData")

# convert list to tensor
xTensor = torch.tensor(xData, dtype=torch.float).to(device)
yTensor = torch.tensor(yData, dtype=torch.float).to(device)
qTensor = torch.empty(0, 1001).to(device=device)
zTensor = torch.empty(0, 2002).to(device=device)
zTNorm  = torch.empty(0, 2002).to(device=device)

# rig hierarchy
parent = CustomFunctions.LoadData("HierarchyData")
parent = [int(i) for i in parent[0]]

if os.path.isfile('../Database/ZNormData.txt'): 
    qData = CustomFunctions.LoadData("QData")
    zData = CustomFunctions.LoadData("ZNormData")
    qTensor = torch.tensor(qData, dtype=torch.float).to(device)
    zTNorm  = torch.tensor(zData, dtype=torch.float).to(device)

    print('Data loaded')

else:
    qTensor = CustomFunctions.ForwardKinematics(yTensor, parent)
    zTensor = torch.cat((yTensor, qTensor), dim=1);
    zTNorm = CustomFunctions.NormalizeData(zTensor)

    # save qData
    with open('../Database/QData.txt', 'w+') as f:
        f.write('\n'.join(' '.join(str(j) for j in i) for i in qTensor.tolist()))
    
    # save zNormData
    with open('../Database/ZNormData.txt', 'w+') as f:
        f.write('\n'.join(' '.join(str(j) for j in i) for i in zTNorm.tolist()))

    print('Data generated')

# dataloader settings for training
train = NNModels.Data.TensorDataset(xTensor, yTensor, qTensor, zTNorm)
train_loader = NNModels.Data.DataLoader(train, shuffle=True, batch_size=32)
c_optimizer, c_scheduler = NNModels.TrainSettings(compressor)
d_optimizer, d_scheduler = NNModels.TrainSettings(decompressor)

# init tensorboard
writer = SummaryWriter()

# ydata pos indexes (erase velocities)
posIdx = []
[posIdx.extend(list(range(i, i + 7))) for i in range(0, yTensor.size(1), 13)]

# train the network. range = epochs
for t in range(10001):
    epochLoss = 0.0

    for batch_idx, (x, y, q, zNorm) in enumerate(train_loader):
        
        # runtime start
        batch_time = time.time()

        x.to(device), y.to(device), zNorm.to(device)

        # autoencoder output: predict/code
        zPred, z_ = compressor(zNorm)

        # Y <- D[X + Code(z_)]
        combinedFeatures = torch.cat((x, z_), dim=1)
        combinedFeatures = CustomFunctions.NormalizeData(combinedFeatures)
        yPred = decompressor(combinedFeatures)
        
        # forward kinematics
        qPred = CustomFunctions.ForwardKinematics(yPred, parent)

        # stats
        print("Batch index: ", batch_idx)
        print("Models + FK runtime:\t %s seconds" % (time.time() - batch_time))

        # losses
        loss_lreg = torch.nn.MSELoss()(zPred[:,:1001], torch.zeros_like(zPred[:,:1001]))
        loss_sreg = torch.nn.L1Loss()(zPred[:,1001:], torch.zeros_like(zPred[:,1001:]))
        loss_loc  = torch.nn.L1Loss()(yPred[:,posIdx], y[:,posIdx])
        loss_chr  = torch.nn.L1Loss()(qPred[:,posIdx], q[:,posIdx])
        loss_lvel = torch.nn.L1Loss()(yPred[:,7:13], y[:,7:13])
        loss_cvel = torch.nn.L1Loss()(qPred[:,7:13], q[:,7:13])

        loss = loss_lreg + loss_sreg + loss_loc + loss_lvel

        c_optimizer.zero_grad() # clear gradients for next train
        d_optimizer.zero_grad() # clear gradients for next train

        loss.backward()         # backpropagation, compute gradients

        c_optimizer.step()      # apply gradients
        d_optimizer.step()      # apply gradients
        c_scheduler.step()      # step learning rate schedule
        d_scheduler.step()      # step learning rate schedule
        
        # free variables
        # del qPred

        # log loss value
        epochLoss += loss * yPred.size(0)

        # stats
        print("Batch runtime:\t\t %s seconds\t\t" % (time.time() - batch_time))
        print("Memory usage:\t\t", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
        print()
        
    # just printing where training is
    # if t % 500 == 0:
    print("Epoch loss", t, epochLoss.item())
    
    # log loss in tensorboard
    # writer.add_scalar('Python Compressor Loss', c_epochLoss, t)
    writer.add_scalar('Python Decompressor Loss', epochLoss, t)

# # saving latent variables (z)
# with open('/home/pau1o-hs/Documents/Database/LatentVariables.txt', "w+") as f:
#     for i in range(len(zTNorm)):
#         zPred, z_ = compressor(zTNorm[i])
#         np.savetxt(f, z_.cpu().detach().numpy()[None], delimiter=" ")

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