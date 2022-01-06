import torch
import numpy as np

device = torch.device("cpu")

def LoadData(filename):
    data = []
    indices = [0]

    with open("database/" + filename + ".txt") as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]

            if inner_list == ['']:
                indices.append(len(data))
                continue

            converted = []
            for item in inner_list: converted.append(float(item))

            data.append(converted)

    data = np.array(data)
    indices = np.array(indices)

    return {
        'data': data,
        'indices': indices
    }

def _fast_cross(a, b):
    return torch.cat([
        a[...,1:2]*b[...,2:3] - a[...,2:3]*b[...,1:2],
        a[...,2:3]*b[...,0:1] - a[...,0:1]*b[...,2:3],
        a[...,0:1]*b[...,1:2] - a[...,1:2]*b[...,0:1]], axis=-1)

def to_xform(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return torch.cat((
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy)), dim=-1)
    
def to_xform_xy(x):

    qw, qx, qy, qz = x[...,0:1], x[...,1:2], x[...,2:3], x[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return torch.cat((
        1.0 - (yy + zz), xy - wz,
        xy + wz, 1.0 - (xx + zz),
        xz - wy, yz + wx,
    ), dim=1)

def from_xy(x):

    c2 = _fast_cross(x[...,0], x[...,1])
    c2 = c2 / torch.sqrt(torch.sum(torch.square(c2), dim=-1))[...,None]
    c1 = _fast_cross(c2, x[...,0])
    c1 = c1 / torch.sqrt(torch.sum(torch.square(c1), dim=-1))[...,None]
    c0 = x[...,0]
    
    # print(torch.cat((
    #     c0[:,0,None], c1[:,0,None], c2[:,0,None],
    #     c0[:,1,None], c1[:,1,None], c2[:,1,None],
    #     c0[:,2,None], c1[:,2,None], c2[:,2,None]
    # ), dim=-1).shape)
    
    return torch.cat((
        c0[...,0,None], c1[...,0,None], c2[...,0,None],
        c0[...,1,None], c1[...,1,None], c2[...,1,None],
        c0[...,2,None], c1[...,2,None], c2[...,2,None]
    ), dim=-1)

def length(x):
    return torch.sqrt(torch.sum(x * x, axis=-1))

def normalize(x, eps=1e-8):
    return x / (length(x)[...,np.newaxis] + eps)

def from_xform(ts):
    
    return normalize(
        torch.where((ts[...,2,2] < 0.0)[...,np.newaxis],
            torch.where((ts[...,0,0] >  ts[...,1,1])[...,np.newaxis],
                torch.cat([
                    (ts[...,2,1]-ts[...,1,2])[...,np.newaxis], 
                    (1.0 + ts[...,0,0] - ts[...,1,1] - ts[...,2,2])[...,np.newaxis], 
                    (ts[...,1,0]+ts[...,0,1])[...,np.newaxis], 
                    (ts[...,0,2]+ts[...,2,0])[...,np.newaxis]], axis=-1),
                torch.cat([
                    (ts[...,0,2]-ts[...,2,0])[...,np.newaxis], 
                    (ts[...,1,0]+ts[...,0,1])[...,np.newaxis], 
                    (1.0 - ts[...,0,0] + ts[...,1,1] - ts[...,2,2])[...,np.newaxis], 
                    (ts[...,2,1]+ts[...,1,2])[...,np.newaxis]], axis=-1)),
            torch.where((ts[...,0,0] < -ts[...,1,1])[...,np.newaxis],
                torch.cat([
                    (ts[...,1,0]-ts[...,0,1])[...,np.newaxis], 
                    (ts[...,0,2]+ts[...,2,0])[...,np.newaxis], 
                    (ts[...,2,1]+ts[...,1,2])[...,np.newaxis], 
                    (1.0 - ts[...,0,0] - ts[...,1,1] + ts[...,2,2])[...,np.newaxis]], axis=-1),
                torch.cat([
                    (1.0 + ts[...,0,0] + ts[...,1,1] + ts[...,2,2])[...,np.newaxis], 
                    (ts[...,2,1]-ts[...,1,2])[...,np.newaxis], 
                    (ts[...,0,2]-ts[...,2,0])[...,np.newaxis], 
                    (ts[...,1,0]-ts[...,0,1])[...,np.newaxis]], axis=-1))))

def from_xform_xy(x):

    c2 = _fast_cross(x[...,0], x[...,1])
    c2 = c2 / torch.sqrt(torch.sum(torch.square(c2), axis=-1))[...,np.newaxis]
    c1 = _fast_cross(c2, x[...,0])
    c1 = c1 / torch.sqrt(torch.sum(torch.square(c1), axis=-1))[...,np.newaxis]
    c0 = x[...,0]

    print(torch.cat([
        c0[...,np.newaxis], 
        c1[...,np.newaxis], 
        c2[...,np.newaxis]], axis=-1))

    return from_xform(torch.cat([
        c0[...,np.newaxis], 
        c1[...,np.newaxis], 
        c2[...,np.newaxis]], axis=-1))

def quat_mul(x, y):
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    return torch.cat([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], dim=-1)

def quat_mul_vec(q, x):
    t = 2.0 * _fast_cross(q[..., 1:], x)
    return x + q[..., 0][..., np.newaxis] * t + _fast_cross(q[..., 1:], t)

def mul(x, y):
    return torch.matmul(x, y)
    
def mul_vec(x, v):
    return torch.matmul(x, v[...,None])[...,0]

def Quat_ForwardKinematics(Y, hierarchy):
    Q = Y[...,:13]
    bone = 1

    for i in range(13, Y.size(1), 13):
        # parent of bone i
        p = hierarchy[bone] * 13

        pos = quat_mul_vec(Q[...,p+3:p+7], Y[...,i+0:i+3]) + (Q[...,p+0:p+3] if p != 0 else 0)
        rot = quat_mul    (Q[...,p+3:p+7], Y[...,i+3:i+7])
        vel = quat_mul_vec(Q[...,p+3:p+7], Y[...,i+7:i+10]) + torch.cross(Q[...,p+10:p+13], quat_mul_vec(Q[...,p+3:p+7], Y[...,i+0:i+3]), dim=-1)  + Q[...,p+12:p+15]
        ang = quat_mul_vec(Q[...,p+3:p+7], Y[...,i+10:i+13]) + Q[...,p+10:p+13]

        Q = torch.cat((Q, pos, rot, vel, ang), dim=1)
        bone += 1

    return Q

def Xform_ForwardKinematics(Y, hierarchy):
    Q = Y[...,:18]
    bone = 1
    
    for i in range(18, Y.size(-1), 18):
        # parent of bone i
        p = hierarchy[bone] * 18

        Qxfm = Q[...,p+3:p+12].reshape(Y.size(0), Y.size(1), 3, 3)

        pos = mul_vec(Qxfm, Y[...,i+0:i+3]) + (Q[...,p+0:p+3] if p != 0 else 0)
        rot = mul    (Qxfm, Y[...,i+3:i+12].reshape(Y.size(0), Y.size(1), 3, 3))
        vel = mul_vec(Qxfm, Y[...,i+12:i+15]) + torch.cross(Q[...,p+15:p+18], mul_vec(Qxfm, Y[...,i+0:i+3]), dim=-1) + Q[...,p+12:p+15]
        ang = mul_vec(Qxfm, Y[...,i+15:i+18]) + Q[...,p+15:p+18]

        Q = torch.cat((Q, pos, rot.flatten(-2), vel, ang), dim=-1)
        bone += 1

    return Q

    # gp, gr, gv, ga = [lpos[...,:1,:]], [lrot[...,:1,:,:]], [lvel[...,:1,:]], [lang[...,:1,:]]
    # for i in range(1, len(parents)):
    #     gp.append(mul_vec(gr[parents[i]], lpos[...,i:i+1,:]) + gp[parents[i]])
    #     gr.append(mul    (gr[parents[i]], lrot[...,i:i+1,:,:]))
    #     gv.append(mul_vec(gr[parents[i]], lvel[...,i:i+1,:]) + 
    #         torch.cross(ga[parents[i]], mul_vec(gr[parents[i]], lpos[...,i:i+1,:]), dim=-1) +
    #         gv[parents[i]])
    #     ga.append(mul_vec(gr[parents[i]], lang[...,i:i+1,:]) + ga[parents[i]])
        
    # return (
    #     torch.cat(gr, dim=-3), 
    #     torch.cat(gp, dim=-2),
    #     torch.cat(gv, dim=-2),
    #     torch.cat(ga, dim=-2))