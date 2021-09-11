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
    def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(Model, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2) 
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
        self.hidden4 = torch.nn.Linear(n_hidden3, n_hidden4)
        self.predict = torch.nn.Linear(n_hidden4, n_output)

    # feed forward
    def forward(self, x):
        x = F.relu(self.hidden(x)) 
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.predict(x)
        return x

# loss function (wip)
def customLoss(noiseInput, predict, target):
    l2loss = torch.nn.L1Loss()(torch.nn.MSELoss()(predict[:, :24], noiseInput), torch.nn.MSELoss()(target[:, :24], noiseInput))
    loss = torch.nn.L1Loss()(predict, target) + l2loss
    return loss

# define nn params
model = Model(n_feature=24, n_hidden=512, n_hidden2=512, n_hidden3=512, n_hidden4=512, n_output=24+32).to(device)

# load data
projectorIn = []
projectorOut = []
latent = []

with open("/home/pau1o-hs/Documents/Database/XData.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]

        if inner_list == ['']:
            continue

        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        projectorIn.append(converted)

with open("/home/pau1o-hs/Documents/Database/LatentVariables.txt") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        
        if inner_list == ['']:
            continue
        
        converted =  np.asarray(inner_list, dtype=np.float64, order='C')
        latent.append(converted)

# combined features (x + z)
projectorOut = np.append(projectorIn, latent, axis=1)

# convert list to tensor
x = torch.tensor(projectorIn, dtype=torch.float).to(device)
y = torch.tensor(projectorOut, dtype=torch.float).to(device)

# normalize data
means = x.mean(dim=1, keepdim=True)
stds = x.std(dim=1, keepdim=True)
normalized_in = (x - means) / stds

means = y.mean(dim=1, keepdim=True)
stds = y.std(dim=1, keepdim=True)
normalized_out = (y - means) / stds

# dataloader settings for training
train = Data.TensorDataset(normalized_in, normalized_out)
train_loader = Data.DataLoader(train, shuffle=True, batch_size=32)

optimizer = optim.RAdam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

# init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(20000):
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
            dist = torch.norm(normalized_in - dataNoise[i], dim=1, p=None)
            knn = dist.topk(1, largest=False)
            knnIndices.append(knn.indices)

        for i in range(0, len(data)):
            newTargets[i] = normalized_out[knnIndices[i]]

        # feed forward
        prediction = model(dataNoise)

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
    writer.add_scalar('Python Projector Loss', epochLoss, t)

# export the model
torch.onnx.export(
    model, torch.rand(1, 24, device=device),
    "projector.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Projector Runtime: %s minutes" % ((time.time() - start_time) / 60))