import sys
sys.path.append('./misc')

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset
import time
import torch
import NNModels
import CustomFunctions
import torch_optimizer as optim
import torch.utils.data as Data

# runtime start
start_time = time.time()

# checking if GPU is available
print(torch.cuda.get_device_name(0)) 
device = torch.device("cuda")

# define nn params
net = NNModels.Stepper(n_feature=24+32, n_hidden=512, n_output=24+32).to(device)

# load data
stepperInput = CustomFunctions.LoadData("XData", True)
latent = CustomFunctions.LoadData("LatentVariables")

# combined features (x + z)
latentCount = 0
for i in range(len(stepperInput)):
    for j in range(len(stepperInput[i])):
        stepperInput[i][j].extend(latent[latentCount])
        latentCount += 1

# output: input shifted -1
stepperNext = stepperInput
for i in range(len(stepperInput)):
    for j in range(len(stepperInput[i])):
        stepperNext[i][j].append(stepperNext[i][j].pop(0))

# normalize data
x = []
y = []
normX = []
normY = []

# list of tensors (each tensor represent a clip)
for i in range(len(stepperInput)):
    x.append(torch.tensor(stepperInput[i], device=device))
    y.append(torch.tensor(stepperNext[i], device=device))
    normX.append(CustomFunctions.NormalizeData(x[i]))
    normY.append(CustomFunctions.NormalizeData(y[i]))

# dataloader settings for training
dataSet = NNModels.StepperDataset(normX, normY)
train_loader = Data.DataLoader(dataSet, shuffle=True, batch_size=32)

optimizer = optim.RAdam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

# init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(1):
    epochLoss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):    
        
        data.to(device), target.to(device)

        # data: (batch size, seq. length)
        # LSTM: (seq. length, batch size, hidden nodes)
        data = torch.transpose(data, 0, 1)
        target = torch.transpose(target, 0, 1)
        n_batch = data.size(1)

        h_t = torch.zeros(1, n_batch, 512, device=device).requires_grad_()
        c_t = torch.zeros(1, n_batch, 512, device=device).requires_grad_()

        # feed forward
        prediction, h_t, c_t = net(data, h_t, c_t)

        # MSELoss: prediction, target
        loss = torch.nn.L1Loss()(prediction, target)   

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
    writer.add_scalar('Python Stepper Loss', epochLoss, t)

# sample input shape
x = torch.rand(1, 1, 56, device=device)
h_t = torch.rand(1, 1, 512, device=device)
c_t = torch.rand(1, 1, 512, device=device)

# export the model
torch.onnx.export(
    net, (x, h_t, c_t),
    "onnx/stepper.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x', 'h0', 'c0'], output_names =['y', 'hn', 'cn']
)

# runtime end
print("Stepper Runtime: %s minutes" % ((time.time() - start_time) / 60))