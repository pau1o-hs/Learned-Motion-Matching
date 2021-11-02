import torch

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
    means = data.mean(dim=dim, keepdim=True)
    stds  = data.std(dim=dim, keepdim=True)
    normalized = (data - means) / stds

    return normalized

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = (mag2)**(0.5)
        v = tuple(n / mag for n in v)
    return v

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
            qt                     = q[:,parent+0 : parent+3] + multComponentA

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