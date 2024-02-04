"""
    Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective

    Xuanmeng Zhang, Minyue Jiang, Zhedong Zheng, Xiao Tan, Errui Ding, Yi Yang

    Project Page : https://github.com/Xuanmeng-Zhang/gnn-re-ranking

    Paper: https://arxiv.org/abs/2012.07620v2

    ======================================================================
   
    On the Market-1501 dataset, we accelerate the re-ranking processing from 89.2s to 9.4ms
    with one K40m GPU, facilitating the real-time post-processing. Similarly, we observe 
    that our method achieves comparable or even better retrieval results on the other four 
    image retrieval benchmarks, i.e., VeRi-776, Oxford-5k, Paris-6k and University-1652, 
    with limited time cost.
"""

import os
import numpy as np
import pickle
import torch


def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pickle(pickle_path, data):
    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def pairwise_squared_distance(x):
    '''
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    '''
    x2s = (x * x).sum(-1, keepdim=True)
    return x2s + x2s.transpose(-1, -2) - 2 * x @ x.transpose(-1, -2)


def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)

    x = x.view(m, -1)
    y = y.view(n, -1)

    dist = torch.pow(x, 2).sum(
        dim=1, keepdim=True
    ).expand(m, n) + torch.pow(y, 2).sum(
        dim=1, keepdim=True
    ).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())

    return dist


def cosine_similarity(x, y):
    m, n = x.size(0), y.size(0)

    x = x.view(m, -1)
    y = y.view(n, -1)

    y = y.t()
    score = torch.mm(x, y)

    return score


def evaluate_ranking_list(
    indices, query_label, query_cam, gallery_label, gallery_cam
):
    CMC = np.zeros((len(gallery_label)), dtype=np.int)
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(
            indices[i], query_label[i], query_cam[i], gallery_label,
            gallery_cam
        )
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.astype(np.float32)
    CMC = CMC / len(query_label) #average CMC
    print(
        'Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' %
        (CMC[0], CMC[4], CMC[9], ap / len(query_label))
    )


def evaluate(index, ql, qc, gl, gc):
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros((len(index)), dtype=np.int)
    if good_index.size == 0: # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i+1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision+precision) / 2

    return ap, cmc
