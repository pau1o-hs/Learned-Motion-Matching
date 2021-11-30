import sys
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
print()

device = torch.device("cuda")

# define nn params
model = NNModels.Projector(n_feature=24, n_hidden=512, n_hidden2=512, n_hidden3=512, n_hidden4=512, n_output=24+32).to(device)

# load data
xData = CustomFunctions.LoadData("XData", False)
zData = CustomFunctions.LoadData("ZData", False)

# list of tensors
xTensor = []
zTensor = []
combinedFeatures = []

xTensor = CustomFunctions.StandardizeData([torch.tensor(xData, dtype=torch.float).to(device)], dim=0)
zTensor.append(torch.tensor(zData, dtype=torch.float).to(device))

# combined features (x + z)
combinedFeatures.append(torch.cat((xTensor[0], zTensor[0]), dim=1))

# relevant dataset indexes
xVal = [list(range(0, 9)) , list(range(12, 15)), list(range(18, 21))]
xTrj = [list(range(0, 9))]
xVal = sum(xVal, [])
xTrj = sum(xTrj, [])

# dataloader settings for training
train = NNModels.Data.TensorDataset(xTensor[0], combinedFeatures[0])
train_loader = NNModels.Data.DataLoader(train, shuffle=True, batch_size=32)
optimizer, scheduler = NNModels.TrainSettings(model)

# training settings
epochs = 1000
logFreq = 500

# init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(epochs + 1):
    epochLoss = 0.0
    xval = 0.0
    zval = 0.0
    dist = 0.0
    epoch_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):

        data.to(device), target.to(device)

        # generate gaussian noise
        dataNoise = torch.zeros_like(data)
        for i in range(data.size(0)):
            dataNoise[i] = data[i] + (np.random.uniform(0.0, 1.0) * torch.randn_like(data[i]))

        # perform knn
        knnIndices = []

        for i in range(data.size(0)):
            diff = torch.norm(xTensor[0] - dataNoise[i], dim=1, p=None)
            knn = diff.topk(1, largest=False)
            knnIndices.append(knn.indices.tolist()[0])
        
        xk          = torch.empty(data.size(0), 0).to(device)
        newTargets  = torch.empty(data.size(0), 0).to(device)

        xk          = torch.cat((xk, xTensor[0][knnIndices]), dim=1)
        newTargets  = torch.cat((newTargets, combinedFeatures[0][knnIndices]), dim=1)

        # feed forward
        prediction = model(dataNoise)

        # losses
        loss_xval = 0.50 * torch.nn.L1Loss()(prediction[:,:24], newTargets[:,:24])  
        loss_zval = 5.00 * torch.nn.L1Loss()(prediction[:,24:], newTargets[:,24:])  
        # loss_dist = 0.02 * torch.nn.L1Loss()(torch.nn.MSELoss()(prediction[:,:24], dataNoise), torch.nn.MSELoss()(xk, dataNoise))  
        loss_dist = 2.00 * torch.nn.L1Loss()(torch.nn.MSELoss()(prediction[:,:24], dataNoise), torch.nn.MSELoss()(xk, dataNoise))

        loss = loss_xval + loss_zval + loss_dist

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()        # step learning rate schedule

        # log loss value
        epochLoss += loss.item()
        xval += loss_xval.item()
        zval += loss_zval.item()
        dist += loss_dist.item()
    
    # just printing where training is
    if t % logFreq == 0:
        print("Epoch:", t, "\tLoss:", epochLoss, "\tRuntime:", (time.time() - epoch_time) * logFreq / 60, "minutes")
        print(xval, zval, dist)
    # log loss in tensorboard  
    writer.add_scalar('Projector Loss', epochLoss, t)

# export the model
torch.onnx.export(
    model, torch.rand(1, 1, 24, device=device),
    "onnx/projector.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Projector Runtime: %s minutes" % ((time.time() - start_time) / 60))