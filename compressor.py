from torch.utils.tensorboard import SummaryWriter
import time
import torch
import numpy as np
import torch_optimizer as optim
import torch.utils.data as Data
import torch.nn.functional as F

# runtime start
start_time = time.time()

# checking if GPU is available
print(torch.cuda.get_device_name(0)) 
device = torch.device("cuda")

# neural network model
class Model(torch.nn.Module):
    # nn layers shape
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_output):
        super(Model, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden) 
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.predict = torch.nn.Linear(n_hidden3, n_output)

    # feed forward
    def forward(self, x, getEncoded=True):
        x = F.elu(self.hidden(x))
        x = F.elu(self.hidden2(x))
        
        # get reduced dimension of data
        if (getEncoded):
            return x

        x = F.elu(self.hidden3(x))
        x = self.predict(x)
        return x

# define nn params
model = Model(n_feature=1232, n_hidden=512, n_hidden2=32, n_hidden3=512, n_output=1232).to(device)

# load data
compressorIn = []
with open("/home/pau1o-hs/Documents/Database/YData.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]

        if inner_list == ['']:
            continue

        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        compressorIn.append(converted)

# convert list to tensor
x = torch.tensor(compressorIn, dtype=torch.float).to(device)
y = torch.tensor(compressorIn, dtype=torch.float).to(device)

# normalize data
means = x.mean(dim=1, keepdim=True)
stds  = x.std(dim=1, keepdim=True)
normalized_data = (x - means) / stds

# dataloader settings for training
train = Data.TensorDataset(normalized_data, normalized_data)
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
        prediction = model(data, False)
        
        # MSELoss: prediction, target
        loss = torch.nn.MSELoss()(prediction, target)     
        
        # clear gradients for next train
        optimizer.zero_grad()

        # backpropagation, compute gradients
        loss.backward()

        # apply gradients
        optimizer.step()

        # step learning rate schedule
        scheduler.step()

        # log loss value
        epochLoss += loss * prediction.size(0)

    # just printing where training is
    if t % 500 == 0:
        print(t, epochLoss.item())
    
    # log loss in tensorboard
    writer.add_scalar('Python Compressor Loss', epochLoss, t)


# saving latent variables (z)
with open('/home/pau1o-hs/Documents/Database/LatentVariables.txt', "w+") as f:
    for i in range(len(normalized_data)):
        prediction = model(normalized_data[i], True)
        np.savetxt(f, prediction.cpu().detach().numpy()[None], delimiter=" ")

# export the model
torch.onnx.export(
    model, torch.rand(1, 1232, device=device), 
    "compressor.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Compressor Runtime: %s minutes" % ((time.time() - start_time) / 60))