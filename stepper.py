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
model = NNModels.Stepper(n_feature=24+32, n_hidden=512, n_output=24+32).to(device)

# load data
xData = CustomFunctions.LoadData("XData", True)
zData = CustomFunctions.LoadData("ZData", True)

# list of tensors
combinedFeaturesI = []
combinedFeaturesN = []

for i in range(len(xData)):
    combinedFeaturesI.append(torch.cat((torch.tensor(xData[i]).to(device), torch.tensor(zData[i]).to(device)), dim=1))
    combinedFeaturesN.append(torch.roll(torch.cat((torch.tensor(xData[i]).to(device), torch.tensor(zData[i]).to(device)), dim=1), -1, 0))

    combinedFeaturesI[i] = CustomFunctions.NormalizeData(combinedFeaturesI[i], dim=1)
    combinedFeaturesN[i] = CustomFunctions.NormalizeData(combinedFeaturesN[i], dim=1)

print()

# dataloader settings for training
train = NNModels.CustomDataset((combinedFeaturesI, combinedFeaturesN), window=20)
train_loader = NNModels.Data.DataLoader(train, shuffle=True, batch_size=32)
optimizer, scheduler = NNModels.TrainSettings(model)

# training settings
epochs = 10000
logFreq = 500

# init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(epochs + 1):
    epochLoss = 0.0
    epoch_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):    
        
        data.to(device), target.to(device)

        # (seq. length, batch size, features) <- (batch size, seq. length, features)
        data    = torch.transpose(data, 0, 1)
        target  = torch.transpose(target, 0, 1)
        n_batch = data.size(1)

        h_t = torch.zeros(1, n_batch, 512, device=device).requires_grad_()
        c_t = torch.zeros(1, n_batch, 512, device=device).requires_grad_()

        # predict x~ and z~ over a window of s frames
        prediction, h_t, c_t = model(data, h_t, c_t)

        # losses
        loss_val = torch.nn.L1Loss()(prediction, target) 
        loss_vel = torch.nn.L1Loss()((prediction[:19] - prediction[1:]) / 0.017, (target[:19] - target[1:]) / 0.017)

        loss = loss_val + loss_vel

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        scheduler.step()        # step learning rate schedule

        # log loss value
        epochLoss += loss.item()

    # just printing where training is
    if t % logFreq == 0:
        print("Epoch:", t, "\tLoss:", epochLoss, "\tRuntime:", (time.time() - epoch_time) * logFreq / 60, "minutes")
    
    # log loss in tensorboard    
    writer.add_scalar('Stepper Loss', epochLoss, t)

# sample input shape
x   = torch.rand(1, 1, 56, device=device)
h_t = torch.rand(1, 1, 512, device=device)
c_t = torch.rand(1, 1, 512, device=device)

# export the model
torch.onnx.export(
    model, (x, h_t, c_t),
    "onnx/stepper.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x', 'h0', 'c0'], output_names =['y', 'hn', 'cn']
)

# runtime end
print("Stepper Runtime: %s minutes" % ((time.time() - start_time) / 60))