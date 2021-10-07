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
    data = (data - means) / stds

    return data

def q_mult(a, b):
    ax, ay, az, aw = a[:,0], a[:,1], a[:,2], a[:,3]
    bx, by, bz, bw = b[:,0], b[:,1], b[:,2], b[:,3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw

    return torch.stack((ox, oy, oz, ow), -1)

def q_conjugate(a):
    x, y, z, w = a[:,0], a[:,1], a[:,2], a[:,3]
    return torch.stack((-x, -y, -z, w), -1)

def qv_mult(q1, v1):
    q2 = torch.cat((v1, torch.zeros(q1.size(0), 1).to(device)), dim=1).to(device)
    output = q_mult(q_mult(q1, q2), q_conjugate(q1))[:,:-1]

    return output

def ForwardKinematics(y, hierarchy):
    # y = yPred.to(device)
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
            qr                    = q[:,parent+3 : parent+7] + multComponentB

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



