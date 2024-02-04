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

import numpy as np
import torch

import gnn_propagate
import build_adjacency_matrix
from utils import *


def gnn_reranking(X_q, X_g, k1, k2):
    query_num, gallery_num = X_q.shape[0], X_g.shape[0]

    X_u = torch.cat((X_q, X_g), axis=0)
    original_score = torch.mm(X_u, X_u.t())
    del X_u, X_q, X_g

    # initial ranking list
    S, initial_rank = original_score.topk(
        k=k1, dim=-1, largest=True, sorted=True
    )

    # stage 1
    A = build_adjacency_matrix.forward(initial_rank.float())
    S = S * S

    # stage 2
    if k2 != 1:
        for i in range(2):
            A = A + A.T
            A = gnn_propagate.forward(
                A, initial_rank[:, :k2].contiguous().float(),
                S[:, :k2].contiguous().float()
            )
            A_norm = torch.norm(A, p=2, dim=1, keepdim=True)
            A = A.div(A_norm.expand_as(A))

    cosine_similarity = torch.mm(A[:query_num, ], A[query_num:, ].t())
    del A, S

    L = torch.sort(-cosine_similarity, dim=1)[1]
    L = L.data.cpu().numpy()
    return L
