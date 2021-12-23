import torch
import numpy as np

device = torch.device("cpu")

# if stepper is true, make a list of clips
def LoadData(filename, isStepper=False):
    preData = []
    data = []
    
    with open("database/" + filename + ".txt") as f:
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
    
    if isStepper == False:
        data = preData

    return data

def NormalizeData(data, dim=1):
    xMin = torch.empty(0, data[0].size(1), device='cuda')
    xMax = torch.empty(0, data[0].size(1), device='cuda')
    normalized = []

    for i in range(len(data)):
        xMin = torch.cat((xMin, torch.min(data[i], 0).values.unsqueeze(0)), dim=0)
        xMax = torch.cat((xMax, torch.max(data[i], 0).values.unsqueeze(0)), dim=0)
    
    xMin = torch.min(xMin, 0).values
    xMax = torch.max(xMax, 0).values
    # print(xMax)

    # normalization
    for i in range(len(data)):
        normalized.append((data[i] - xMin) / (xMax - xMin))

    return normalized

def StandardizeData(data, dim=1):
    mean = 0
    std = 0
    count = 0
    normalized = []

    # mean
    for i in range(len(data)):
        mean = mean + data[i].sum(dim=dim)
        count = count + data[i].size(0)

    mean = mean / count
    
    # std
    for i in range(len(data)):
        std = std + ((data[i] - mean) ** 2).sum(dim=dim)
    
    std = ((std / count) + 0.001) ** 0.5
    
    # for i in range(len(std)): 
        # if std[i] == 0:
            # std[i] = 1
            # if data[0][0,i] == mean[i]: print(i)

    # standardization
    for i in range(len(data)):
        normalized.append((data[i] - mean) / (std))

    return normalized

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[:,0]
    q1 = Q[:,1]
    q2 = Q[:,2]
    q3 = Q[:,3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = torch.stack((r00, r01, r02, r10, r11, r12, r20, r21, r22), 1)

    return rot_matrix

def q_mult(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw

    return torch.stack((ow, ox, oy, oz), -1)
    # quaternions = torch.stack((ow, ox, oy, oz), -1)
    # return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def q_conjugate(a):
    w, x, y, z = a[:,0], a[:,1], a[:,2], a[:,3]
    return torch.stack((w, -x, -y, -z), -1)

def qv_mult(q1, v1):
    q2 = torch.cat((torch.zeros(q1.size(0), 1).to(device), v1), dim=1).to(device)
    output = q_mult(q_mult(q1, q2), q_conjugate(q1))[:,1:]

    return output

def ForwardKinematics(y, hierarchy):
    output = []

    for i in range(len(y)):
        q = torch.empty(y[i].size(0), 0).to(device)
        q = torch.cat((q, y[i][:,:13].to(device)), dim=1).to(device)
        bone = 1

        for j in range(13, y[i].size(1), 13):
            yt      = y[i][:,j:j+3].to(device)
            yr      = y[i][:,j+3:j+7].to(device)
            ytVel   = y[i][:,j+7:j+10].to(device)
            yrVel   = y[i][:,j+10:j+13].to(device)
            
            parent = hierarchy[bone] * 13

            # qt
            multComponentA = qv_mult(q[:,parent+3 : parent+7], yt)

            if parent != 0:
                qt = q[:,parent+0 : parent+3] + multComponentA
            else:
                qt = multComponentA

            # qr
            multComponentB = q_mult(q[:,parent+3 : parent+7], yr)
            qr                    = multComponentB

            # qrVel
            multComponentC = qv_mult(q[:,parent+3  : parent+7], yrVel)
            qrVel                  = q[:,parent+10 : parent+13] + multComponentC
            
            # qtVel
            multComponentD = qv_mult(q[:,parent+3 : parent+7], ytVel)
            qtVel                  = q[:,parent+7 : parent+10] + multComponentD + torch.cross(qrVel, multComponentA, dim=1)

            bone += 1
            q = torch.cat((q, qt, qr, qtVel, qrVel), dim=1).to(device)

        output.append(q.to('cuda'))

    return output