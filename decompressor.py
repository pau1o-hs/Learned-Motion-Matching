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
device = torch.device("cpu")

# load data
X = CustomFunctions.LoadData("XData")['data']
Y = CustomFunctions.LoadData("YData")['data']
indices = CustomFunctions.LoadData("YData")['indices']

hierarchy = CustomFunctions.LoadData("HierarchyData")['data']
hierarchy = [int(hierarchy[i][0]) for i in range(len(hierarchy))]

# To tensor
X = torch.as_tensor(X, dtype=torch.float).to(device)
Y = torch.as_tensor(Y, dtype=torch.float).to(device)

# compute forward kinematics
Q = CustomFunctions.Quat_ForwardKinematics(Y, hierarchy)

# convert to rotation/two-column matrix
Yxfm = torch.empty((Y.shape[0], 0), dtype=torch.float).to(device)
Ytxy = torch.empty((Y.shape[0], 0), dtype=torch.float).to(device)

Qxfm = torch.empty((Q.shape[0], 0), dtype=torch.float).to(device)
Qtxy = torch.empty((Q.shape[0], 0), dtype=torch.float).to(device)

Yrtd = torch.empty((Y.shape[0], 0), dtype=torch.float).to(device)
Qrtd = torch.empty((Q.shape[0], 0), dtype=torch.float).to(device)

for i in range(0, Y.size(1), 13):
    Yxfm = torch.cat((Yxfm, Y[:,i:i+3], CustomFunctions.to_xform(Y[:,i+3:i+7]), Y[:,i+7:i+13]), dim=-1)
    Qxfm = torch.cat((Qxfm, Q[:,i:i+3], CustomFunctions.to_xform(Q[:,i+3:i+7]), Q[:,i+7:i+13]), dim=-1)
    Ytxy = torch.cat((Ytxy, Y[:,i:i+3], CustomFunctions.to_xform_xy(Y[:,i+3:i+7]), Y[:,i+7:i+13]), dim=-1)
    Qtxy = torch.cat((Qtxy, Q[:,i:i+3], CustomFunctions.to_xform_xy(Q[:,i+3:i+7]), Q[:,i+7:i+13]), dim=-1)

# relevant dataset indexes
pos = []
rot = []
vel = []
ang = []
posXfm = []
rotXfm = []
velXfm = []
angXfm = []

[pos.extend(list(range(i + 0, i + 3)))  for i in range(0, Ytxy.size(1), 15)]
[rot.extend(list(range(i + 3, i + 9)))  for i in range(0, Ytxy.size(1), 15)]
[vel.extend(list(range(i + 9, i + 12)))  for i in range(0, Ytxy.size(1), 15)]
[ang.extend(list(range(i + 12, i + 15)))  for i in range(0, Ytxy.size(1), 15)]
[posXfm.extend(list(range(i + 0, i + 3)))  for i in range(18, Qxfm.size(1), 18)]
[rotXfm.extend(list(range(i + 3, i + 12)))  for i in range(18, Qxfm.size(1), 18)]
[velXfm.extend(list(range(i + 12, i + 15)))  for i in range(18, Qxfm.size(1), 18)]
[angXfm.extend(list(range(i + 15, i + 18)))  for i in range(18, Qxfm.size(1), 18)]

# scales
Ypos_scale = Ytxy[:,pos].std()
Ytxy_scale = Ytxy[:,rot].std()
Yvel_scale = Ytxy[:,vel].std()
Yang_scale = Ytxy[:,ang].std()

Qpos_scale = Qtxy[:,pos].std()
Qtxy_scale = Qtxy[:,rot].std()
Qvel_scale = Qtxy[:,vel].std()
Qang_scale = Qtxy[:,ang].std()

# means and stds
compressor_mean = torch.cat((Ytxy, Qtxy), dim=1).mean(dim=0)
compressor_std = torch.empty((0))

for i in range(0, Ytxy.size(1), 15):
    compressor_std = torch.cat((compressor_std, Ypos_scale.repeat(3), Ytxy_scale.repeat(6), Yvel_scale.repeat(3), Yang_scale.repeat(3)), dim=0)

for i in range(0, Qtxy.size(1), 15):
    compressor_std = torch.cat((compressor_std, Qpos_scale.repeat(3), Qtxy_scale.repeat(6), Qvel_scale.repeat(3), Qang_scale.repeat(3)), dim=0)

torch.set_printoptions(profile="full", precision=8)
print(Ytxy[:,pos].sum(), Ypos_scale, Ytxy_scale, Yvel_scale, Yang_scale)

decompressor_mean = Ytxy.mean(dim=0)
decompressor_std = Ytxy.std(dim=0) + 0.001

# Training settings
nfeatures = X.size(1)
nlatent = 32
epochs = 10000
batchsize=32
window=2
logFreq = 500
dt = 1.0 / 60.0

compressor   = NNModels.Compressor(Ytxy.size(1) * 2, nlatent).to(device)
decompressor = NNModels.Decompressor(nfeatures + nlatent, Ytxy.size(1)).to(device)

# Dataloader settings
# train = NNModels.CustomDataset(datas=(X, Ytxy, Qtxy, Qxfm), indices=indices, window=2)
# train_loader = NNModels.Data.DataLoader(train, shuffle=True, batch_size=32)
c_optimizer, c_scheduler = NNModels.TrainSettings(compressor)
d_optimizer, d_scheduler = NNModels.TrainSettings(decompressor)

# Build batches respecting window size
samples = []
for i in range(len(indices)):
    for j in range(indices[i] - window):
        samples.append(np.arange(j, j + window))
samples = torch.as_tensor(np.array(samples))

# Init tensorboard
writer = SummaryWriter()

# train the network. range = epochs
for t in range(epochs + 1):
    epoch_time = time.time()

    # batch
    batch = samples[torch.randint(0, len(samples), size=[batchsize])]
    # print(X[batch.long()])
    # (seq. length, batch size, features) <- (batch size, seq. length, features)
    Xgnd = X[batch.long()].transpose(0, 1)
    Ygnd = Ytxy[batch.long()].transpose(0, 1)
    Qgnd = Qtxy[batch.long()].transpose(0, 1)
    Qgnd_xfm = Qxfm[batch.long()].transpose(0, 1)
    
    # Generate latent variables Z
    Zgnd = compressor((torch.cat((Ygnd, Qgnd), dim=-1) - compressor_mean) / compressor_std)

    # Reconstruct pose Y
    Ytil = decompressor(torch.cat((Xgnd, Zgnd), dim=-1)) * decompressor_std + decompressor_mean

    # Recompute forward kinematics
    Ytil_xfm = torch.empty((Ytil.size(0), Ytil.size(1), 0)).to(device)

    for i in range(0, Ytil.size(2), 15):
        Ytil_xfm = torch.cat((Ytil_xfm, Ytil[...,i:i+3], CustomFunctions.from_xy(Ytil[...,i+3:i+9].reshape(Ytil.size(0), Ytil.size(1), 3, 2)), Ytil[...,i+9:i+15]), dim=-1)

    Qtil_xfm = CustomFunctions.Xform_ForwardKinematics(Ytil_xfm, hierarchy)

    # Compute deltas
    Ygnd_dlt = (Ygnd[1] - Ygnd[0]) / dt
    Ytil_dlt = (Ytil[1] - Ytil[0]) / dt
    
    Qgnd_dlt = (Qgnd_xfm[1] - Qgnd_xfm[0]) / dt
    Qtil_dlt = (Qtil_xfm[1] - Qtil_xfm[0]) / dt

    Zgnd_dvel = (Zgnd[1] - Zgnd[0]) / dt

    # Latent regularization losses
    loss_lreg = torch.mean(100. * torch.square(Zgnd))
    loss_sreg = torch.mean(100. * torch.abs(Zgnd))
    loss_vreg = torch.mean(10.0 * torch.abs(Zgnd_dvel))

    # Local & character space losses
    loss_loc_pos  = torch.mean(10000 * torch.abs(Ygnd[:,:,pos] - Ytil[:,:,pos]))
    loss_loc_txy  = torch.mean(7000 * torch.abs(Ygnd[:,:,rot] - Ytil[:,:,rot]))
    loss_loc_vel  = torch.mean(7000 * torch.abs(Ygnd[:,:,vel] - Ytil[:,:,vel]))
    loss_loc_ang  = torch.mean(1.25 * torch.abs(Ygnd[:,:,ang] - Ytil[:,:,ang]))

    loss_chr_pos  = torch.mean(10000 * torch.abs(Qgnd_xfm[:,:,posXfm] - Qtil_xfm[:,:,posXfm]))
    loss_chr_xfm  = torch.mean(10000 * torch.abs(Qgnd_xfm[:,:,rotXfm] - Qtil_xfm[:,:,rotXfm]))
    loss_chr_vel  = torch.mean(0.35 * torch.abs(Qgnd_xfm[:,:,velXfm] - Qtil_xfm[:,:,velXfm]))
    loss_chr_ang  = torch.mean(0.35 * torch.abs(Qgnd_xfm[:,:,angXfm] - Qtil_xfm[:,:,angXfm]))

    # local & character space velocity losses
    loss_lvel_pos = torch.mean(1000 * torch.abs(Ygnd_dlt[:,pos] - Ytil_dlt[:,pos]))
    loss_lvel_rot = torch.mean(175 * torch.abs(Ygnd_dlt[:,rot] - Ytil_dlt[:,rot]))

    loss_cvel_pos = torch.mean(200. * torch.abs(Qgnd_dlt[:,posXfm] - Qtil_dlt[:,posXfm]))
    loss_cvel_rot = torch.mean(75.0 * torch.abs(Qgnd_dlt[:,rotXfm] - Qtil_dlt[:,rotXfm]))

    loss = (
    # loss_lreg + loss_sreg + loss_vreg + 
    loss_loc_pos + loss_loc_txy + loss_loc_vel + loss_loc_ang +
    loss_chr_pos + loss_chr_xfm + loss_chr_vel + loss_chr_ang +
    loss_lvel_pos + loss_lvel_rot + loss_cvel_pos + loss_cvel_rot
    )

    # clear gradients for next train
    c_optimizer.zero_grad()
    d_optimizer.zero_grad()

    # backpropagation, compute gradients
    loss.backward()

    # apply gradients
    c_optimizer.step()
    d_optimizer.step()

    # step learning rate schedule
    if t % 1000 == 0:
        c_scheduler.step()
        d_scheduler.step()

    writer.add_scalar('decompressor/loss', loss.item(), t)
    
    writer.add_scalars('decompressor/loss_terms', {
        'lreg': loss_lreg.item(),
        'sreg': loss_sreg.item(),
        'vreg': loss_vreg.item(),
        'loc_pos': loss_loc_pos.item(),
        'loc_txy': loss_loc_txy.item(),
        'loc_vel': loss_loc_vel.item(),
        'loc_ang': loss_loc_ang.item(),
        'chr_pos': loss_chr_pos.item(),
        'chr_xfm': loss_chr_xfm.item(),
        'chr_vel': loss_chr_vel.item(),
        'chr_ang': loss_chr_ang.item(),
        'lvel_pos': loss_lvel_pos.item(),
        'lvel_rot': loss_lvel_rot.item(),
        'cvel_pos': loss_cvel_pos.item(),
        'cvel_rot': loss_cvel_rot.item(),
    }, t)
    
    writer.add_scalars('decompressor/latent', {
        'mean': Zgnd.mean().item(),
        'std': Zgnd.std().item(),
    }, t)

    # just printing where training is
    if t % logFreq == 0:
        print("Epoch:", t, "\tLoss:", loss.item(), "\tRuntime:", (time.time() - epoch_time) * logFreq / 60, "minutes")

# save character space transforms (Ytxy)
with open('database/YtxyData.txt', "w+") as f:
    for i in range(Ytxy.size(0)):
        np.savetxt(f, Ytxy[i].cpu().detach().numpy()[None], delimiter=" ")

        if (i + 1) in indices:
            np.savetxt(f, [''], fmt='%s')

# save character space transforms (Qtxy)
with open('database/QtxyData.txt', "w+") as f:
    for i in range(Qtxy.size(0)):
        np.savetxt(f, Qtxy[i].cpu().detach().numpy()[None], delimiter=" ")

        if (i + 1) in indices:
            np.savetxt(f, [''], fmt='%s')

# save latent variables (Z)
with open('database/ZData.txt', "w+") as f:
    for i in range(Ytxy.size(0)):
        Zgnd = compressor((torch.cat((Ytxy[i], Qtxy[i]), dim=-1) - compressor_mean) / compressor_std)
        np.savetxt(f, Zgnd.cpu().detach().numpy()[None], delimiter=" ")
        
        if (i + 1) in indices:
            np.savetxt(f, [''], fmt='%s')

# export compressor model
torch.onnx.export(
    compressor, torch.rand(1, 1, Ytxy.size(1) * 2, device=device), 
    "onnx/compressor.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['z']
)

# export decompressor model
torch.onnx.export(
    decompressor, torch.rand(1, 1, nfeatures + nlatent, device=device), 
    "onnx/decompressor.onnx", export_params=True,
    opset_version=9, do_constant_folding=True,
    input_names = ['x'], output_names = ['y']
)

# runtime end
print("Decompressor runtime: %s minutes" % ((time.time() - start_time) / 60))