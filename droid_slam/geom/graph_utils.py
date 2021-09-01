
import torch
import numpy as np
from collections import OrderedDict

import lietorch
from data_readers.rgbd_utils import compute_distance_matrix_flow, compute_distance_matrix_flow2


def graph_to_edge_list(graph):
    ii, jj, kk = [], [], []
    for s, u in enumerate(graph):
        for v in graph[u]:
            ii.append(u)
            jj.append(v)
            kk.append(s)

    ii = torch.as_tensor(ii)
    jj = torch.as_tensor(jj)
    kk = torch.as_tensor(kk)
    return ii, jj, kk

def keyframe_indicies(graph):
    return torch.as_tensor([u for u in graph])

def meshgrid(m, n, device='cuda'):
    ii, jj = torch.meshgrid(torch.arange(m), torch.arange(n))
    return ii.reshape(-1).to(device), jj.reshape(-1).to(device)

def neighbourhood_graph(n, r):
    ii, jj = meshgrid(n, n)
    d = (ii - jj).abs()
    keep = (d >= 1) & (d <= r)
    return ii[keep], jj[keep]


def build_frame_graph(poses, disps, intrinsics, num=16, thresh=24.0, r=2):
    """ construct a frame graph between co-visible frames """
    N = poses.shape[1]
    poses = poses[0].cpu().numpy()
    disps = disps[0][:,3::8,3::8].cpu().numpy()
    intrinsics = intrinsics[0].cpu().numpy() / 8.0
    d = compute_distance_matrix_flow(poses, disps, intrinsics)

    count = 0
    graph = OrderedDict()
    
    for i in range(N):
        graph[i] = []
        d[i,i] = np.inf
        for j in range(i-r, i+r+1):
            if 0 <= j < N and i != j:
                graph[i].append(j)
                d[i,j] = np.inf
                count += 1

    while count < num:
        ix = np.argmin(d)
        i, j = ix // N, ix % N

        if d[i,j] < thresh:
            graph[i].append(j)
            d[i,j] = np.inf
            count += 1
        else:
            break
    
    return graph



def build_frame_graph_v2(poses, disps, intrinsics, num=16, thresh=24.0, r=2):
    """ construct a frame graph between co-visible frames """
    N = poses.shape[1]
    # poses = poses[0].cpu().numpy()
    # disps = disps[0].cpu().numpy()
    # intrinsics = intrinsics[0].cpu().numpy()
    d = compute_distance_matrix_flow2(poses, disps, intrinsics)

    # import matplotlib.pyplot as plt
    # plt.imshow(d)
    # plt.show()

    count = 0
    graph = OrderedDict()
    
    for i in range(N):
        graph[i] = []
        d[i,i] = np.inf
        for j in range(i-r, i+r+1):
            if 0 <= j < N and i != j:
                graph[i].append(j)
                d[i,j] = np.inf
                count += 1

    while 1:
        ix = np.argmin(d)
        i, j = ix // N, ix % N

        if d[i,j] < thresh:
            graph[i].append(j)

            for i1 in range(i-1, i+2):
                for j1 in range(j-1, j+2):
                    if 0 <= i1 < N and 0 <= j1 < N:
                        d[i1, j1] = np.inf

            count += 1
        else:
            break
    
    return graph

