# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from __future__ import print_function
import numpy as np
from libc.stdint cimport int64_t, uint64_t

import cython

cimport numpy as np

import random
from collections import defaultdict

"""
Compiler directives:
https://github.com/cython/cython/wiki/enhancements-compilerdirectives

Cython tutorial:
https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html

Credit to https://github.com/luzai
"""


# Main interface
cpdef evaluate_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03=False):
    distmat = np.asarray(distmat, dtype=np.float32)
    q_pids = np.asarray(q_pids, dtype=np.int64)
    g_pids = np.asarray(g_pids, dtype=np.int64)
    q_camids = np.asarray(q_camids, dtype=np.int64)
    g_camids = np.asarray(g_camids, dtype=np.int64)
    if use_metric_cuhk03:
        return eval_cuhk03_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    return eval_market1501_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)


cpdef eval_cuhk03_cy(float[:,:] distmat, int64_t[:] q_pids, int64_t[:]g_pids,
                     int64_t[:]q_camids, int64_t[:]g_camids, int64_t max_rank):

    cdef int64_t num_q = distmat.shape[0]
    cdef int64_t num_g = distmat.shape[1]

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    cdef:
        int64_t num_repeats = 10
        int64_t[:,:] indices = np.argsort(distmat, axis=1)
        int64_t[:,:] matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.int64)

        float[:,:] all_cmc = np.zeros((num_q, max_rank), dtype=np.float32)
        float[:] all_AP = np.zeros(num_q, dtype=np.float32)
        float num_valid_q = 0. # number of valid query

        int64_t q_idx, q_pid, q_camid, g_idx
        int64_t[:] order = np.zeros(num_g, dtype=np.int64)
        int64_t keep

        float[:] raw_cmc = np.zeros(num_g, dtype=np.float32) # binary vector, positions with value 1 are correct matches
        float[:] masked_raw_cmc = np.zeros(num_g, dtype=np.float32)
        float[:] cmc, masked_cmc
        int64_t num_g_real, num_g_real_masked, rank_idx, rnd_idx
        uint64_t meet_condition
        float AP
        int64_t[:] kept_g_pids, mask

        float num_rel
        float[:] tmp_cmc = np.zeros(num_g, dtype=np.float32)
        float tmp_cmc_sum

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        for g_idx in range(num_g):
            order[g_idx] = indices[q_idx, g_idx]
        num_g_real = 0
        meet_condition = 0
        kept_g_pids = np.zeros(num_g, dtype=np.int64)

        for g_idx in range(num_g):
            if (g_pids[order[g_idx]] != q_pid) or (g_camids[order[g_idx]] != q_camid):
                raw_cmc[num_g_real] = matches[q_idx][g_idx]
                kept_g_pids[num_g_real] = g_pids[order[g_idx]]
                num_g_real += 1
                if matches[q_idx][g_idx] > 1e-31:
                    meet_condition = 1

        if not meet_condition:
            # this condition is true when query identity does not appear in gallery
            continue

        # cuhk03-specific setting
        g_pids_dict = defaultdict(list) # overhead!
        for g_idx in range(num_g_real):
            g_pids_dict[kept_g_pids[g_idx]].append(g_idx)

        cmc = np.zeros(max_rank, dtype=np.float32)
        for _ in range(num_repeats):
            mask = np.zeros(num_g_real, dtype=np.int64)

            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                #rnd_idx = idxs[0] # use deterministic for debugging
                mask[rnd_idx] = 1

            num_g_real_masked = 0
            for g_idx in range(num_g_real):
                if mask[g_idx] == 1:
                    masked_raw_cmc[num_g_real_masked] = raw_cmc[g_idx]
                    num_g_real_masked += 1

            masked_cmc = np.zeros(num_g, dtype=np.float32)
            function_cumsum(masked_raw_cmc, masked_cmc, num_g_real_masked)
            for g_idx in range(num_g_real_masked):
                if masked_cmc[g_idx] > 1:
                    masked_cmc[g_idx] = 1

            for rank_idx in range(max_rank):
                cmc[rank_idx] += masked_cmc[rank_idx] / num_repeats

        for rank_idx in range(max_rank):
            all_cmc[q_idx, rank_idx] = cmc[rank_idx]
        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        function_cumsum(raw_cmc, tmp_cmc, num_g_real)
        num_rel = 0
        tmp_cmc_sum = 0
        for g_idx in range(num_g_real):
            tmp_cmc_sum += (tmp_cmc[g_idx] / (g_idx + 1.)) * raw_cmc[g_idx]
            num_rel += raw_cmc[g_idx]
        all_AP[q_idx] = tmp_cmc_sum / num_rel
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    # compute averaged cmc
    cdef float[:] avg_cmc = np.zeros(max_rank, dtype=np.float32)
    for rank_idx in range(max_rank):
        for q_idx in range(num_q):
            avg_cmc[rank_idx] += all_cmc[q_idx, rank_idx]
        avg_cmc[rank_idx] /= num_valid_q

    cdef float mAP = 0
    for q_idx in range(num_q):
        mAP += all_AP[q_idx]
    mAP /= num_valid_q

    return np.asarray(avg_cmc).astype(np.float32), mAP


cpdef eval_market1501_cy(float[:,:] distmat, int64_t[:] q_pids, int64_t[:]g_pids,
                         int64_t[:]q_camids, int64_t[:]g_camids, int64_t max_rank):

    cdef int64_t num_q = distmat.shape[0]
    cdef int64_t num_g = distmat.shape[1]

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    cdef:
        int64_t[:,:] indices = np.argsort(distmat, axis=1)
        int64_t[:,:] matches = (np.asarray(g_pids)[np.asarray(indices)] == np.asarray(q_pids)[:, np.newaxis]).astype(np.int64)

        float[:,:] all_cmc = np.zeros((num_q, max_rank), dtype=np.float32)
        float[:] all_AP = np.zeros(num_q, dtype=np.float32)
        float num_valid_q = 0. # number of valid query

        int64_t q_idx, q_pid, q_camid, g_idx
        int64_t[:] order = np.zeros(num_g, dtype=np.int64)
        int64_t keep

        float[:] raw_cmc = np.zeros(num_g, dtype=np.float32) # binary vector, positions with value 1 are correct matches
        float[:] cmc = np.zeros(num_g, dtype=np.float32)
        int64_t num_g_real, rank_idx
        uint64_t meet_condition

        float num_rel
        float[:] tmp_cmc = np.zeros(num_g, dtype=np.float32)
        float tmp_cmc_sum

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        for g_idx in range(num_g):
            order[g_idx] = indices[q_idx, g_idx]
        num_g_real = 0
        meet_condition = 0

        for g_idx in range(num_g):
            if (g_pids[order[g_idx]] != q_pid) or (g_camids[order[g_idx]] != q_camid):
                raw_cmc[num_g_real] = matches[q_idx][g_idx]
                num_g_real += 1
                if matches[q_idx][g_idx] > 1e-31:
                    meet_condition = 1

        if not meet_condition:
            # this condition is true when query identity does not appear in gallery
            continue

        # compute cmc
        function_cumsum(raw_cmc, cmc, num_g_real)
        for g_idx in range(num_g_real):
            if cmc[g_idx] > 1:
                cmc[g_idx] = 1

        for rank_idx in range(max_rank):
            all_cmc[q_idx, rank_idx] = cmc[rank_idx]
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        function_cumsum(raw_cmc, tmp_cmc, num_g_real)
        num_rel = 0
        tmp_cmc_sum = 0
        for g_idx in range(num_g_real):
            tmp_cmc_sum += (tmp_cmc[g_idx] / (g_idx + 1.)) * raw_cmc[g_idx]
            num_rel += raw_cmc[g_idx]
        all_AP[q_idx] = tmp_cmc_sum / num_rel

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    # compute averaged cmc
    cdef float[:] avg_cmc = np.zeros(max_rank, dtype=np.float32)
    for rank_idx in range(max_rank):
        for q_idx in range(num_q):
            avg_cmc[rank_idx] += all_cmc[q_idx, rank_idx]
        avg_cmc[rank_idx] /= num_valid_q

    cdef float mAP = 0
    for q_idx in range(num_q):
        mAP += all_AP[q_idx]
    mAP /= num_valid_q

    return np.asarray(avg_cmc).astype(np.float32), mAP


# Compute the cumulative sum
cdef void function_cumsum(cython.numeric[:] src, cython.numeric[:] dst, int64_t n):
    cdef int64_t i
    dst[0] = src[0]
    for i in range(1, n):
        dst[i] = src[i] + dst[i - 1]