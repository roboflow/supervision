# Person Attribute Recognition
This code was developed for the experiment of person attribute recognition in [Omni-Scale Feature Learning for Person Re-Identification (ICCV'19)](https://arxiv.org/abs/1905.00953).

## Download data
Download the PA-100K dataset from [https://github.com/xh-liu/HydraPlus-Net](https://github.com/xh-liu/HydraPlus-Net), and extract the file under the folder where you store your data (say $DATASET). The folder structure should look like
```bash
$DATASET/
    pa100k/
        data/ # images
        annotation/
            annotation.mat
```

## Train
The training command is provided in `train.sh`. Run `bash train.sh $DATASET` to start training.

## Test
To test a pretrained model, add the following two arguments to `train.sh`: `--load-weights $PATH_TO_WEIGHTS --evaluate`.