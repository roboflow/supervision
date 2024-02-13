from __future__ import print_function

import os.path as osp
import sys
import timeit


sys.path.insert(0, osp.dirname(osp.abspath(__file__)) + "/../../..")
"""
Test the speed of cython-based evaluation code. The speed improvements
can be much bigger when using the real reid data, which contains a larger
amount of query and gallery images.

Note: you might encounter the following error:
  'AssertionError: Error: all query identities do not appear in gallery'.
This is normal because the inputs are random numbers. Just try again.
"""

print("*** Compare running time ***")

setup = """
import sys
import os.path as osp
import numpy as np
sys.path.insert(0, osp.dirname(osp.abspath(__file__)) + '/../../..')
from supervision.tracker.stronsort_tracker.torchreid import metrics
num_q = 30
num_g = 300
max_rank = 5
distmat = np.random.rand(num_q, num_g) * 20
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)
q_camids = np.random.randint(0, 5, size=num_q)
g_camids = np.random.randint(0, 5, size=num_g)
"""

print("=> Using market1501's metric")
pytime = timeit.timeit(
    "metrics.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=False)",
    setup=setup,
    number=20,
)
cytime = timeit.timeit(
    "metrics.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=True)",
    setup=setup,
    number=20,
)
print("Python time: {} s".format(pytime))
print("Cython time: {} s".format(cytime))
print("Cython is {} times faster than python\n".format(pytime / cytime))

print("=> Using cuhk03's metric")
pytime = timeit.timeit(
    "metrics.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03=True, use_cython=False)",
    setup=setup,
    number=20,
)
cytime = timeit.timeit(
    "metrics.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03=True, use_cython=True)",
    setup=setup,
    number=20,
)
print("Python time: {} s".format(pytime))
print("Cython time: {} s".format(cytime))
print("Cython is {} times faster than python\n".format(pytime / cytime))
"""
print("=> Check precision")

num_q = 30
num_g = 300
max_rank = 5
distmat = np.random.rand(num_q, num_g) * 20
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)
q_camids = np.random.randint(0, 5, size=num_q)
g_camids = np.random.randint(0, 5, size=num_g)

cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=False)
print("Python:\nmAP = {} \ncmc = {}\n".format(mAP, cmc))
cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=True)
print("Cython:\nmAP = {} \ncmc = {}\n".format(mAP, cmc))
"""
