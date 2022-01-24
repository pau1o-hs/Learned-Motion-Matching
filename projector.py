import sys
sys.path.append('./misc')

from torch.utils.tensorboard import SummaryWriter
import time
import torch
import NNModels
import CustomFunctions
import numpy as np
from sklearn.neighbors import BallTree

# runtime start
start_time = time.time()

# checking if GPU is available
print(torch.cuda.get_device_name(0))
print()

device = torch.device("cpu")

# Load data
X = CustomFunctions.LoadData("XData")['data']
Z = CustomFunctions.LoadData("ZData")['data']

# To tensor
X = torch.as_tensor(X, dtype=torch.float).to(device)
Z = torch.as_tensor(Z, dtype=torch.float).to(device)

# means and stds
projector_mean_in = X.mean(dim=0)
projector_std_in  = X.std().repeat(X.size(1))

projector_mean_out = torch.cat((X, Z), dim=-1).mean(dim=0)
projector_std_out  = torch.cat((X, Z), dim=-1).std(dim=0) + 0.001

# training settings
nframes = X.size(0)
nfeatures = X.size(1)
nlatent = 32
epochs = 500000
batchsize = 32
logFreq = 500

projector = NNModels.Projector(nfeatures, nfeatures + nlatent).to(device)

# dataloader settings for training
optimizer, scheduler = NNModels.TrainSettings(projector)

# Fit acceleration structure for nearest neighbors search    
tree = BallTree(X)

# init tensorboard
writer = SummaryWriter()

X_noise_std = X.std(dim=0) + 1.0

# train the network. range = epochs
for t in range(epochs + 1):
    epoch_time = time.time()

    # Batch
    batch = torch.randint(0, nframes, size=[batchsize])

    nsigma = np.random.uniform(size=[batchsize, 1]).astype(np.float32)
    noise = np.random.normal(size=[batchsize, nfeatures]).astype(np.float32)
    Xhat = X[batch] + X_noise_std * nsigma * noise

    # Find nearest
    nearest = tree.query(Xhat, k=1, return_distance=False)[:,0]
    
    Xgnd = torch.as_tensor(X[nearest])
    Zgnd = torch.as_tensor(Z[nearest])
    Xhat = torch.as_tensor(Xhat)
    Dgnd = torch.sqrt(torch.sum(torch.square(Xhat - Xgnd), dim=-1))
    
    # Projector
    pred = (projector((Xhat - projector_mean_in) / projector_std_in) *
        projector_std_out + projector_mean_out)
    
    Xtil = pred[:,:nfeatures]
    Ztil = pred[:,nfeatures:]
    Dtil = torch.sqrt(torch.sum(torch.square(Xhat - Xtil), dim=-1))
    
    # Compute Losses
    loss_xval = torch.mean(1.00 * torch.abs(Xgnd - Xtil))
    loss_zval = torch.mean(0.15 * torch.abs(Zgnd - Ztil))
    loss_dist = torch.mean(1.00 * torch.abs(Dgnd - Dtil))
    
    loss = loss_xval + loss_zval + loss_dist

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    
    # Step learning rate schedule
    if t % 1000 == 0:
        scheduler.step()

    # just printing where training is
    if t % logFreq == 0:
        print("Epoch:", t, "\tLoss:", loss.item(), "\tRuntime:", (time.time() - epoch_time) * logFreq / 60, "minutes")
        torch.set_printoptions(profile="full", precision=8)
        print(Xgnd[0,0], Xtil[0,0])

    # log loss in tensorboard
    writer.add_scalar('projector/loss', loss.item(), t)

    writer.add_scalars('projector/loss_terms', {
        'xval': loss_xval.item(),
        'zval': loss_zval.item(),
        'dist': loss_dist.item(),
    }, t)

# export the model
torch.onnx.export(
    projector, torch.rand(1, 1, 24, device=device),
    "onnx/projector.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Projector Runtime: %s minutes" % ((time.time() - start_time) / 60))