import sys
sys.path.append('./misc')

from torch.utils.tensorboard import SummaryWriter
import time
import torch
import NNModels
import CustomFunctions

# runtime start
start_time = time.time()

# checking if GPU is available
print(torch.cuda.get_device_name(0)) 
device = torch.device("cuda")

# define nn params
model = NNModels.Stepper(n_feature=24+32, n_hidden=512, n_hidden2=512, n_output=24+32).to(device)

# load data
xData = CustomFunctions.LoadData("XData", True)
zData = CustomFunctions.LoadData("ZData", True)

# list of tensors
xTensor = []
zTensor = []

for i in range(len(xData)):
    xTensor.append(torch.tensor(xData[i], dtype=torch.float).to(device))
    zTensor.append(torch.tensor(zData[i], dtype=torch.float).to(device))

xTensor = CustomFunctions.StandardizeData(xTensor, dim=0)

# list of tensors
combinedFeaturesI = []
combinedFeaturesN = []
combinedFeaturesD = []

for i in range(len(xData)):
    combinedFeaturesI.append(torch.cat((xTensor[i], zTensor[i]), dim=1))
    combinedFeaturesN.append(torch.roll(torch.cat((xTensor[i], zTensor[i]), dim=1), -1, 0))

    # combinedFeaturesI[i] = CustomFunctions.NormalizeData(combinedFeaturesI[i], dim=0)
    # combinedFeaturesN[i] = CustomFunctions.NormalizeData(combinedFeaturesN[i], dim=0)
    combinedFeaturesD.append(combinedFeaturesN[i] - combinedFeaturesI[i])

print()

# relevant dataset indexes
xVal = [list(range(0, 9)) , list(range(12, 15)), list(range(18, 21))]
xVel = [list(range(9, 12)), list(range(15, 18)), list(range(21, 24))]
xVal = sum(xVal, [])
xVel = sum(xVel, [])

# dataloader settings for training
train = NNModels.CustomDataset((combinedFeaturesI, combinedFeaturesD, combinedFeaturesN), window=20)
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
    xvel = 0.0
    zvel = 0.0
    epoch_time = time.time()

    for batch_idx, (data, delta, target) in enumerate(train_loader):    
        
        data.to(device), target.to(device)

        # (seq. length, batch size, features) <- (batch size, seq. length, features)
        data    = torch.transpose(data, 0, 1)
        target  = torch.transpose(target, 0, 1)

        pred = torch.empty(0, data.size(1), 56).to(device)
        pred = torch.cat((pred, data[1].view(1, data.size(1), 56)), dim=0)

        # predict delta x and delta z over a window of s frames
        for i in range(1, 20):
            prediction = model(pred[i - 1])
            pred = torch.cat((pred, (pred[i - 1] + prediction).view(1, data.size(1), 56)), dim=0)
        
        # losses
        loss_xval = 2.000 * torch.nn.L1Loss()(pred[:,:,:24], target[:,:,:24])
        loss_zval = 1.000 * torch.nn.L1Loss()(pred[:,:,24:], target[:,:,24:])

        # loss_xvel = 0.050 * torch.nn.L1Loss()(pred[:,:,xVel], target[:,:,xVel])
        loss_xvel = 0.050 * torch.nn.L1Loss()((pred[1:,:,:24] - pred[:19,:,:24]) / 0.017, (target[1:,:,:24] - target[:19,:,:24]) / 0.017)
        loss_zvel = 0.015 * torch.nn.L1Loss()((pred[1:,:,24:] - pred[:19,:,24:]) / 0.017, (target[1:,:,24:] - target[:19,:,24:]) / 0.017)

        loss = loss_xval + loss_zval + loss_xvel + loss_zvel
        # loss = loss_xval + loss_zval

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()        # step learning rate schedule

        # log loss value
        epochLoss += loss.item()
        xval += loss_xval.item()
        zval += loss_zval.item()
        xvel += loss_xvel.item()
        zvel += loss_zvel.item()

    # just printing where training is
    if t % logFreq == 0:
        print("Epoch:", t, "\tLoss:", epochLoss, "\tRuntime:", (time.time() - epoch_time) * logFreq / 60, "minutes")
        print(xval, zval, xvel, zvel)
    
    # log loss in tensorboard    
    writer.add_scalar('Stepper Loss', epochLoss, t)

# sample input shape
x  = torch.rand(1, 1, 56, device=device)

# export the model
torch.onnx.export(
    model, x,
    "onnx/stepper.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names =['y']
)

# runtime end
print("Stepper Runtime: %s minutes" % ((time.time() - start_time) / 60))