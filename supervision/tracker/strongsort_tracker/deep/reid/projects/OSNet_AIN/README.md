# Differentiable NAS for OSNet-AIN

## Introduction
This repository contains the neural architecture search (NAS) code (based on [Torchreid](https://arxiv.org/abs/1910.10093)) for [OSNet-AIN](https://arxiv.org/abs/1910.06827), an extension of [OSNet](https://arxiv.org/abs/1905.00953) that achieves strong performance on cross-domain person re-identification (re-ID) benchmarks (*without using any target data*). OSNet-AIN builds on the idea of using [instance normalisation](https://arxiv.org/abs/1607.08022) (IN) layers to eliminate instance-specific contrast in images for domain-generalisable representation learning. This is inspired by the [neural style transfer](https://arxiv.org/abs/1703.06868) works that use IN to remove image styles. Though IN naturally suits the cross-domain person re-ID task, it still remains unclear that where to insert IN to a re-ID CNN can maximise the performance gain. To avoid exhaustively evaluating all possible designs, OSNet-AIN learns to search for the optimal OSNet+IN design from data using a differentiable NAS algorithm. For technical details, please refer to our paper at https://arxiv.org/abs/1910.06827.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1yvVIi2Ml7WBe85Uhaa54qyG4g8z-MGEB" width="500px" />
</div>

## Training
Assume the reid data is stored at `$DATA`. Run
```
python main.py --config-file nas.yaml --root $DATA
```

The structure of the found architecture will be shown at the end of training.

The default config was designed for 8 Tesla V100 32GB GPUs. You can modify the batch size based on your device memory.

**Note** that the test result obtained at the end of architecture search is not meaningful (due to the stochastic sampling layers). Therefore, do not rely on the result to judge the model performance. Instead, you should construct the found architecture in `osnet_child.py` and re-train and evaluate the model on the reid datasets.

## Citation
If you find this code useful to your research, please consider citing the following papers.
```
@article{zhou2021osnet,
  title={Learning Generalisable Omni-Scale Representations for Person Re-Identification},
  author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  journal={TPAMI},
  year={2021}
}

@inproceedings{zhou2019osnet,
  title={Omni-Scale Feature Learning for Person Re-Identification},
  author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  booktitle={ICCV},
  year={2019}
}
```