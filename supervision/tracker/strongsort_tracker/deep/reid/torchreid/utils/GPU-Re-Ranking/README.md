# Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective

[[Paper]](https://arxiv.org/abs/2012.07620v2)

On the Market-1501 dataset, we accelerate the re-ranking processing from **89.2s** to **9.4ms** with one K40m GPU, facilitating the real-time post-processing. 
Similarly, we observe that our method achieves comparable or even better retrieval results on the other four image retrieval benchmarks, 
i.e., VeRi-776, Oxford-5k, Paris-6k and University-1652, with limited time cost.

## Prerequisites

The code was mainly developed and tested with python 3.7, PyTorch 1.4.1, CUDA 10.2, and CentOS release 6.10.

The code has been included in `/extension`. To compile it:

```shell
cd extension
sh make.sh
```

## Demo

The demo script `main.py` provides the gnn re-ranking  method using the prepared feature. 

```shell
python main.py --data_path PATH_TO_DATA --k1 26 --k2 7
```

## Citation
```bibtex
@article{zhang2020understanding,
  title={Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective},
  author={Xuanmeng Zhang, Minyue Jiang, Zhedong Zheng, Xiao Tan, Errui Ding, Yi Yang},
  journal={arXiv preprint arXiv:2012.07620},
  year={2020}
}
```

