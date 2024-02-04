# Model Zoo

- Results are presented in the format of *<Rank-1 (mAP)>*.
- When computing model size and FLOPs, only layers that are used at test time are considered (see `torchreid.utils.compute_model_complexity`).
- Asterisk (\*) means the model is trained from scratch.
- `combineall=True` means all images in the dataset are used for model training.
- Why not use heavy data augmentation like [random erasing](https://arxiv.org/abs/1708.04896) for model training? It's because heavy data augmentation might harm the cross-dataset generalization performance (see [this paper](https://arxiv.org/abs/1708.04896)).


## ImageNet pretrained models


| Model | Download |
| :--- | :---: |
| shufflenet | [model](https://drive.google.com/file/d/1RFnYcHK1TM-yt3yLsNecaKCoFO4Yb6a-/view?usp=sharing) |
| mobilenetv2_x1_0 | [model](https://drive.google.com/file/d/1K7_CZE_L_Tf-BRY6_vVm0G-0ZKjVWh3R/view?usp=sharing) |
| mobilenetv2_x1_4 | [model](https://drive.google.com/file/d/10c0ToIGIVI0QZTx284nJe8QfSJl5bIta/view?usp=sharing) |
| mlfn | [model](https://drive.google.com/file/d/1PP8Eygct5OF4YItYRfA3qypYY9xiqHuV/view?usp=sharing) |
| osnet_x1_0 | [model](https://drive.google.com/file/d/1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY/view?usp=sharing) |
| osnet_x0_75 | [model](https://drive.google.com/file/d/1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq/view?usp=sharing) |
| osnet_x0_5 | [model](https://drive.google.com/file/d/16DGLbZukvVYgINws8u8deSaOqjybZ83i/view?usp=sharing) |
| osnet_x0_25 | [model](https://drive.google.com/file/d/1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs/view?usp=sharing) |
| osnet_ibn_x1_0 | [model](https://drive.google.com/file/d/1sr90V6irlYYDd4_4ISU2iruoRG8J__6l/view?usp=sharing) |
| osnet_ain_x1_0 | [model](https://drive.google.com/file/d/1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo/view?usp=sharing) |
| osnet_ain_x0_75 | [model](https://drive.google.com/file/d/1apy0hpsMypqstfencdH-jKIUEFOW4xoM/view?usp=sharing) |
| osnet_ain_x0_5 | [model](https://drive.google.com/file/d/1KusKvEYyKGDTUBVRxRiz55G31wkihB6l/view?usp=sharing) |
| osnet_ain_x0_25 | [model](https://drive.google.com/file/d/1SxQt2AvmEcgWNhaRb2xC4rP6ZwVDP0Wt/view?usp=sharing) |


## Same-domain ReID


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance | market1501  | dukemtmcreid | msmt17 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| resnet50 | 23.5 | 2.7 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [87.9 (70.4)](https://drive.google.com/file/d/1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV/view?usp=sharing) | [78.3 (58.9)](https://drive.google.com/file/d/17ymnLglnc64NRvGOitY3BqMRS9UWd1wg/view?usp=sharing) | [63.2 (33.9)](https://drive.google.com/file/d/1ep7RypVDOthCRIAqDnn4_N-UhkkFHJsj/view?usp=sharing) |
| resnet50_fc512 | 24.6 | 4.1 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [90.8 (75.3)](https://drive.google.com/file/d/1kv8l5laX_YCdIGVCetjlNdzKIA3NvsSt/view?usp=sharing) | [81.0 (64.0)](https://drive.google.com/file/d/13QN8Mp3XH81GK4BPGXobKHKyTGH50Rtx/view?usp=sharing) | [69.6 (38.4)](https://drive.google.com/file/d/1fDJLcz4O5wxNSUvImIIjoaIF9u1Rwaud/view?usp=sharing) |
| mlfn | 32.5 | 2.8 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [90.1 (74.3)](https://drive.google.com/file/d/1wXcvhA_b1kpDfrt9s2Pma-MHxtj9pmvS/view?usp=sharing) | [81.1 (63.2)](https://drive.google.com/file/d/1rExgrTNb0VCIcOnXfMsbwSUW1h2L1Bum/view?usp=sharing) | [66.4 (37.2)](https://drive.google.com/file/d/18JzsZlJb3Wm7irCbZbZ07TN4IFKvR6p-/view?usp=sharing) |
| hacnn<sup>*</sup> | 4.5 | 0.5 | softmax | (160, 64) | `random_flip`, `random_crop` | `euclidean` | [90.9 (75.6)](https://drive.google.com/file/d/1LRKIQduThwGxMDQMiVkTScBwR7WidmYF/view?usp=sharing) | [80.1 (63.2)](https://drive.google.com/file/d/1zNm6tP4ozFUCUQ7Sv1Z98EAJWXJEhtYH/view?usp=sharing) | [64.7 (37.2)](https://drive.google.com/file/d/1MsKRtPM5WJ3_Tk2xC0aGOO7pM3VaFDNZ/view?usp=sharing) |
| mobilenetv2_x1_0 | 2.2 | 0.2 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [85.6 (67.3)](https://drive.google.com/file/d/18DgHC2ZJkjekVoqBWszD8_Xiikz-fewp/view?usp=sharing) | [74.2 (54.7)](https://drive.google.com/file/d/1q1WU2FETRJ3BXcpVtfJUuqq4z3psetds/view?usp=sharing) | [57.4 (29.3)](https://drive.google.com/file/d/1j50Hv14NOUAg7ZeB3frzfX-WYLi7SrhZ/view?usp=sharing) |
| mobilenetv2_x1_4 | 4.3 | 0.4 | softmax | (256, 128) | `random_flip`, `random_crop` | `euclidean` | [87.0 (68.5)](https://drive.google.com/file/d/1t6JCqphJG-fwwPVkRLmGGyEBhGOf2GO5/view?usp=sharing) | [76.2 (55.8)](https://drive.google.com/file/d/12uD5FeVqLg9-AFDju2L7SQxjmPb4zpBN/view?usp=sharing) | [60.1 (31.5)](https://drive.google.com/file/d/1ZY5P2Zgm-3RbDpbXM0kIBMPvspeNIbXz/view?usp=sharing) |
| osnet_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip` | `euclidean` | [94.2 (82.6)](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view?usp=sharing) | [87.0 (70.2)](https://drive.google.com/file/d/1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq/view?usp=sharing) | [74.9 (43.8)](https://drive.google.com/file/d/112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M/view?usp=sharing) |
| osnet_x0_75 | 1.3 | 0.57 | softmax | (256, 128) | `random_flip` | `euclidean` | [93.7 (81.2)](https://drive.google.com/file/d/1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer/view?usp=sharing) | [85.8 (69.8)](https://drive.google.com/file/d/1IE3KRaTPp4OUa6PGTFL_d5_KQSJbP0Or/view?usp=sharing) | [72.8 (41.4)](https://drive.google.com/file/d/1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc/view?usp=sharing) |
| osnet_x0_5 | 0.6 | 0.27 | softmax | (256, 128) | `random_flip` | `euclidean` | [92.5 (79.8)](https://drive.google.com/file/d/1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT/view?usp=sharing) | [85.1 (67.4)](https://drive.google.com/file/d/1KoUVqmiST175hnkALg9XuTi1oYpqcyTu/view?usp=sharing) | [69.7 (37.5)](https://drive.google.com/file/d/1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv/view?usp=sharing) |
| osnet_x0_25 | 0.2 | 0.08 | softmax | (256, 128) | `random_flip` | `euclidean` | [91.2 (75.0)](https://drive.google.com/file/d/1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj/view?usp=sharing) | [82.0 (61.4)](https://drive.google.com/file/d/1eumrtiXT4NOspjyEV4j8cHmlOaaCGk5l/view?usp=sharing) | [61.4 (29.5)](https://drive.google.com/file/d/1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF/view?usp=sharing) |


## Cross-domain ReID

#### Market1501 -> DukeMTMC-reID


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance  | Rank-1 | Rank-5 | Rank-10 | mAP | Download |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| osnet_ibn_x1_0 | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 48.5 | 62.3 | 67.4 | 26.7 | [model](https://drive.google.com/file/d/1uWW7_z_IcUmRNPqQOrEBdsvic94fWH37/view?usp=sharing) |
| osnet_ain_x1_0 | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `cosine` | 52.4 | 66.1 | 71.2 | 30.5 | [model](https://drive.google.com/file/d/14bNFGm0FhwHEkEpYKqKiDWjLNhXywFAd/view?usp=sharing) |


#### DukeMTMC-reID -> Market1501


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance  | Rank-1 | Rank-5 | Rank-10 | mAP | Download |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| osnet_ibn_x1_0 | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 57.7 | 73.7 | 80.0 | 26.1 | [model](https://drive.google.com/file/d/1CNxL1IP0BjcE1TSttiVOID1VNipAjiF3/view?usp=sharing) |
| osnet_ain_x1_0 | 2.2 | 0.98  | softmax | (256, 128) | `random_flip`, `color_jitter` | `cosine` | 61.0 | 77.0 | 82.5 | 30.6 | [model](https://drive.google.com/file/d/1hypJvq8G04SOby6jvF337GEkg5K_bmCw/view?usp=sharing) |


#### MSMT17 (`combineall=True`) -> Market1501 & DukeMTMC-reID


| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance | msmt17 -> market1501 | msmt17 -> dukemtmcreid | Download |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: |
| resnet50 | 23.5 | 2.7 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 46.3 (22.8) | 52.3 (32.1) | [model](https://drive.google.com/file/d/1yiBteqgIZoOeywE8AhGmEQl7FTVwrQmf/view?usp=sharing) |
| osnet_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 66.6 (37.5) | 66.0 (45.3) | [model](https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view?usp=sharing) |
| osnet_x0_75 | 1.3 | 0.57 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 63.6 (35.5) | 65.3 (44.5) | [model](https://drive.google.com/file/d/1fhjSS_7SUGCioIf2SWXaRGPqIY9j7-uw/view?usp=sharing) |
| osnet_x0_5 | 0.6 | 0.27 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 64.3 (34.9) | 65.2 (43.3) | [model](https://drive.google.com/file/d/1DHgmb6XV4fwG3n-CnCM0zdL9nMsZ9_RF/view?usp=sharing) |
| osnet_x0_25 | 0.2 | 0.08 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 59.9 (31.0) | 61.5 (39.6) | [model](https://drive.google.com/file/d/1Kkx2zW89jq_NETu4u42CFZTMVD5Hwm6e/view?usp=sharing) |
| osnet_ibn_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `euclidean` | 66.5 (37.2) | 67.4 (45.6) | [model](https://drive.google.com/file/d/1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ/view?usp=sharing) |
| osnet_ain_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `cosine` | 70.1 (43.3) | 71.1 (52.7) | [model](https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view?usp=sharing) |


#### Multi-source domain generalization

The models below are trained using multiple source datasets, as described in [Zhou et al. TPAMI'21](https://arxiv.org/abs/1910.06827).

Regarding the abbreviations, MS is MSMT17; M is Market1501; D is DukeMTMC-reID; and C is CUHK03.

All models were trained with [im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine.yaml](https://github.com/KaiyangZhou/deep-person-reid/blob/master/configs/im_osnet_ain_x1_0_softmax_256x128_amsgrad_cosine.yaml) and `max_epoch=50`.

| Model | # Param (10^6) | GFLOPs | Loss | Input | Transforms | Distance | MS+D+C->M | MS+M+C->D | MS+D+M->C |D+M+C->MS |
| :--- | :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: | :---: |
| osnet_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `cosine` | [72.5 (44.2)](https://drive.google.com/file/d/1tuYY1vQXReEd8N8_npUkc7npPDDmjNCV/view?usp=sharing) | [65.2 (47.0)](https://drive.google.com/file/d/1UxUI4NsE108UCvcy3O1Ufe73nIVPKCiu/view?usp=sharing) | [23.9 (23.3)](https://drive.google.com/file/d/1kAA6qHJvbaJtyh1b39ZyEqWROwUgWIhl/view?usp=sharing) | [33.2 (12.6)](https://drive.google.com/file/d/1wAHuYVTzj8suOwqCNcEmu6YdbVnHDvA2/view?usp=sharing) |
| osnet_ibn_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `cosine` | [73.0 (44.9)](https://drive.google.com/file/d/14sH6yZwuNHPTElVoEZ26zozOOZIej5Mf/view?usp=sharing) | [64.6 (45.7)](https://drive.google.com/file/d/1Sk-2SSwKAF8n1Z4p_Lm_pl0E6v2WlIBn/view?usp=sharing) | [25.7 (25.4)](https://drive.google.com/file/d/1actHP7byqWcK4eBE1ojnspSMdo7k2W4G/view?usp=sharing) | [39.8 (16.2)](https://drive.google.com/file/d/1BGOSdLdZgqHe2qFafatb-5sPY40JlYfp/view?usp=sharing) |
| osnet_ain_x1_0 | 2.2 | 0.98 | softmax | (256, 128) | `random_flip`, `color_jitter` | `cosine` | [73.3 (45.8)](https://drive.google.com/file/d/1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB/view?usp=sharing) | [65.6 (47.2)](https://drive.google.com/file/d/1YjJ1ZprCmaKG6MH2P9nScB9FL_Utf9t1/view?usp=sharing) | [27.4 (27.1)](https://drive.google.com/file/d/1IxIg5P0cei3KPOJQ9ZRWDE_Mdrz01ha2/view?usp=sharing) | [40.2 (16.2)](https://drive.google.com/file/d/1KcoUKzLmsUoGHI7B6as_Z2fXL50gzexS/view?usp=sharing) |
