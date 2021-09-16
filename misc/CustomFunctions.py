import numpy as np
import torch
from torch._C import device

# if stepper is true, make a list of clips
def LoadData(filename, isStepper=False):
    preData = []
    data = []
    
    with open("/home/pau1o-hs/Documents/Database/" + filename + ".txt") as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]

            if inner_list == ['']:
                if isStepper:
                    data.append(preData)
                    preData = []
                continue

            converted = []
            for item in inner_list: converted.append(float(item))

            preData.append(converted)
    
    if not isStepper:
        data = preData

    return data

def NormalizeData(data):
    means = data.mean(dim=1, keepdim=True)
    stds  = data.std(dim=1, keepdim=True)
    normalized_data = (data - means) / stds

    return normalized_data

def q_mult(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw

    return torch.stack((ow, ox, oy, oz), -1)

def q_conjugate(a):
    w, x, y, z = torch.unbind(a, -1)
    return torch.stack((w, -x, -y, -z), -1)

def qv_mult(q1, v1):
    q2 = torch.cat((torch.tensor([0.0], device='cuda'), v1), dim=0)
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def ForwardKinematics(y, parent):
    output = torch.empty(0, y.size(1), device='cuda', requires_grad=True)

    for i in range(y.size(0)):
        q = torch.empty(0, 13, device='cuda', requires_grad=True)
        q = torch.cat((q, y[i][:13].view(1, 13)), dim=0)

        current = 1

        for j in range(13, y.size(1), 13):
            yt      = y[i,j:j+3]
            yr      = y[i,j+3:j+7]
            ytVel   = y[i,j+7:j+10]
            yrVel   = y[i,j+10:j+13]
            
            multComponentA = qv_mult(q[parent[current]][3:7], yt)
            qt      = q[parent[current]][0:3] + multComponentA

            multComponentB = q_mult(q[parent[current]][3:7], yr)
            qr      = q[parent[current]][3:7] + multComponentB

            multComponentC = qv_mult(q[parent[current]][3:7], yrVel)
            qrVel   = q[parent[current]][10:13] + multComponentC
            
            multComponentD = qv_mult(q[parent[current]][3:7], ytVel)
            qtVel   = q[parent[current]][7:10] + multComponentD + torch.cross(qrVel, multComponentA, dim=0)

            current += 1
            q = torch.cat((q, torch.cat((qt, qr, qtVel, qrVel), dim=0).view(1, 13)), dim=0)

        qAppend = torch.flatten(q)
        output = torch.cat((output, qAppend.view(1, qAppend.size(0))))
        
        del q
        del qAppend

    return output



