# Deep mutual learning

This repo implements [Deep Mutual Learning (CVPR'18)](https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf) (DML) for person re-id.

We used this code in our [OSNet](https://arxiv.org/pdf/1905.00953.pdf) paper (see Supp. B). The training command to reproduce the result of "triplet + DML" (Table 12f in the paper) is
```bash
python main.py \
--config-file im_osnet_x1_0_dml_256x128_amsgrad_cosine.yaml \
--root $DATA
```

`$DATA` corresponds to the path to your dataset folder.

Change `model.deploy` to `both` if you wanna enable model ensembling.

If you have any questions, please raise an issue in the Issues area.