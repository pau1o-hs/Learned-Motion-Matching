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
device = torch.device("cuda")

# define nn params
model = NNModels.Projector(n_feature=24, n_hidden=512, n_hidden2=512, n_hidden3=512, n_hidden4=512, n_output=24+32).to(device)

# load data
projectorIn = CustomFunctions.LoadData("XData")
latent = CustomFunctions.LoadData("LatentVariables")

# combined features (x + z)
projectorOut = np.append(projectorIn, latent, axis=1)

# convert list to tensor
x = torch.tensor(projectorIn, dtype=torch.float).to(device)
y = torch.tensor(projectorOut, dtype=torch.float).to(device)

# normalize data
normX = CustomFunctions.NormalizeData(x)
normY = CustomFunctions.NormalizeData(y)

# dataloader settings for training
train = NNModels.Data.TensorDataset(normX, normY)
train_loader = NNModels.Data.DataLoader(train, shuffle=True, batch_size=32)
optimizer, scheduler = NNModels.TrainSettings(model)

# init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(1):
    epochLoss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):

        data.to(device), target.to(device)

        # generate gaussian noise
        dataNoise = torch.zeros_like(data).to(device)

        for i in range(data.size(0)):
            dataNoise[i] = data[i] + torch.randn_like(data[i]).to(device) * np.random.uniform(0.0, 1.0)

        # perform knn
        knnIndices = []
        newTargets = torch.zeros(target.size(0), target.size(1)).to(device)

        for i in range(0, len(data)):
            dist = torch.norm(normX - dataNoise[i], dim=1, p=None)
            knn = dist.topk(1, largest=False)
            knnIndices.append(knn.indices)

        for i in range(0, len(data)):
            newTargets[i] = normY[knnIndices[i]]

        # feed forward
        prediction = model(dataNoise)

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
    writer.add_scalar('Python Projector Loss', epochLoss, t)

# export the model
torch.onnx.export(
    model, torch.rand(1, 24, device=device),
    "onnx/projector.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Projector Runtime: %s minutes" % ((time.time() - start_time) / 60))