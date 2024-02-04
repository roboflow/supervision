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
import argparse
import torch

from utils import *
from gnn_reranking import *

parser = argparse.ArgumentParser(description='Reranking_is_GNN')
parser.add_argument(
    '--data_path',
    type=str,
    default='../xm_rerank_gpu_2/features/market_88_test.pkl',
    help='path to dataset'
)
parser.add_argument(
    '--k1',
    type=int,
    default=26, # Market-1501
    # default=60,   # Veri-776
    help='parameter k1'
)
parser.add_argument(
    '--k2',
    type=int,
    default=7, # Market-1501
    # default=10,   # Veri-776
    help='parameter k2'
)

args = parser.parse_args()


def main():
    data = load_pickle(args.data_path)

    query_cam = data['query_cam']
    query_label = data['query_label']
    gallery_cam = data['gallery_cam']
    gallery_label = data['gallery_label']

    gallery_feature = torch.FloatTensor(data['gallery_f'])
    query_feature = torch.FloatTensor(data['query_f'])
    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    indices = gnn_reranking(query_feature, gallery_feature, args.k1, args.k2)
    evaluate_ranking_list(
        indices, query_label, query_cam, gallery_label, gallery_cam
    )


if __name__ == '__main__':
    main()
