Torchreid
===========
Torchreid is a library for deep-learning person re-identification, written in `PyTorch <https://pytorch.org/>`_ and developed for our ICCV'19 project, `Omni-Scale Feature Learning for Person Re-Identification <https://arxiv.org/abs/1905.00953>`_.

It features:

- multi-GPU training
- support both image- and video-reid
- end-to-end training and evaluation
- incredibly easy preparation of reid datasets
- multi-dataset training
- cross-dataset evaluation
- standard protocol used by most research papers
- highly extensible (easy to add models, datasets, training methods, etc.)
- implementations of state-of-the-art deep reid models
- access to pretrained reid models
- advanced training techniques
- visualization tools (tensorboard, ranks, etc.)


Code: https://github.com/KaiyangZhou/deep-person-reid.

Documentation: https://kaiyangzhou.github.io/deep-person-reid/.

How-to instructions: https://kaiyangzhou.github.io/deep-person-reid/user_guide.

Model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.

Tech report: https://arxiv.org/abs/1910.10093.

You can find some research projects that are built on top of Torchreid `here <https://github.com/KaiyangZhou/deep-person-reid/tree/master/projects>`_.


What's new
------------
- [Aug 2021] We have released the ImageNet-pretrained models of ``osnet_ain_x0_75``, ``osnet_ain_x0_5`` and ``osnet_ain_x0_25``. The pretraining setup follows `pycls <https://github.com/facebookresearch/pycls/blob/master/configs/archive/imagenet/resnet/R-50-1x64d_step_8gpu.yaml>`_.
- [Apr 2021] We have updated the appendix in the `TPAMI version of OSNet <https://arxiv.org/abs/1910.06827v5>`_ to include results in the multi-source domain generalization setting. The trained models can be found in the `Model Zoo <https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html>`_.
- [Apr 2021] We have added a script to automate the process of calculating average results over multiple splits. For more details please see ``tools/parse_test_res.py``.
- [Apr 2021] ``v1.4.0``: We added the person search dataset, `CUHK-SYSU <http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html>`_.  Please see the `documentation <https://kaiyangzhou.github.io/deep-person-reid/>`_ regarding how to download the dataset (it contains cropped person images).
- [Apr 2021] All models in the model zoo have been moved to google drive. Please raise an issue if any model's performance is inconsistent with the numbers shown in the model zoo page (could be caused by wrong links).
- [Mar 2021] `OSNet <https://arxiv.org/abs/1910.06827>`_ will appear in the TPAMI journal! Compared with the conference version, which focuses on discriminative feature learning using the omni-scale building block, this journal extension further considers generalizable feature learning by integrating `instance normalization layers <https://arxiv.org/abs/1607.08022>`_ with the OSNet architecture. We hope this journal paper can motivate more future work to taclke the generalization issue in cross-dataset re-ID.
- [Mar 2021] Generalization across domains (datasets) in person re-ID is crucial in real-world applications, which is closely related to the topic of *domain generalization*. Interested in learning how the field of domain generalization has developed over the last decade? Check our recent survey in this topic at https://arxiv.org/abs/2103.02503, with coverage on the history, datasets, related problems, methodologies, potential directions, and so on (*methods designed for generalizable re-ID are also covered*!).
- [Feb 2021] ``v1.3.6`` Added `University-1652 <https://dl.acm.org/doi/abs/10.1145/3394171.3413896>`_, a new dataset for multi-view multi-source geo-localization (credit to `Zhedong Zheng <https://github.com/layumi>`_).
- [Feb 2021] ``v1.3.5``: Now the `cython code <https://github.com/KaiyangZhou/deep-person-reid/pull/412>`_ works on Windows (credit to `lablabla <https://github.com/lablabla>`_).
- [Jan 2021] Our recent work, `MixStyle <https://openreview.net/forum?id=6xHJ37MVxxp>`_ (mixing instance-level feature statistics of samples of different domains for improving domain generalization), has been accepted to ICLR'21. The code has been released at https://github.com/KaiyangZhou/mixstyle-release where the person re-ID part is based on Torchreid.
- [Jan 2021] A new evaluation metric called `mean Inverse Negative Penalty (mINP)` for person re-ID has been introduced in `Deep Learning for Person Re-identification: A Survey and Outlook (TPAMI 2021) <https://arxiv.org/abs/2001.04193>`_. Their code can be accessed at `<https://github.com/mangye16/ReID-Survey>`_.
- [Aug 2020] ``v1.3.3``: Fixed bug in ``visrank`` (caused by not unpacking ``dsetid``).
- [Aug 2020] ``v1.3.2``: Added ``_junk_pids`` to ``grid`` and ``prid``. This avoids using mislabeled gallery images for training when setting ``combineall=True``.
- [Aug 2020] ``v1.3.0``: (1) Added ``dsetid`` to the existing 3-tuple data source, resulting in ``(impath, pid, camid, dsetid)``. This variable denotes the dataset ID and is useful when combining multiple datasets for training (as a dataset indicator). E.g., when combining ``market1501`` and ``cuhk03``, the former will be assigned ``dsetid=0`` while the latter will be assigned ``dsetid=1``. (2) Added ``RandomDatasetSampler``. Analogous to ``RandomDomainSampler``, ``RandomDatasetSampler`` samples a certain number of images (``batch_size // num_datasets``) from each of specified datasets (the amount is determined by ``num_datasets``).
- [Aug 2020] ``v1.2.6``: Added ``RandomDomainSampler`` (it samples ``num_cams`` cameras each with ``batch_size // num_cams`` images to form a mini-batch).
- [Jun 2020] ``v1.2.5``: (1) Dataloader's output from ``__getitem__`` has been changed from ``list`` to ``dict``. Previously, an element, e.g. image tensor, was fetched with ``imgs=data[0]``. Now it should be obtained by ``imgs=data['img']``. See this `commit <https://github.com/KaiyangZhou/deep-person-reid/commit/aefe335d68f39a20160860e6d14c2d34f539b8a5>`_ for detailed changes. (2) Added ``k_tfm`` as an option to image data loader, which allows data augmentation to be applied ``k_tfm`` times *independently* to an image. If ``k_tfm > 1``, ``imgs=data['img']`` returns a list with ``k_tfm`` image tensors.
- [May 2020] Added the person attribute recognition code used in `Omni-Scale Feature Learning for Person Re-Identification (ICCV'19) <https://arxiv.org/abs/1905.00953>`_. See ``projects/attribute_recognition/``.
- [May 2020] ``v1.2.1``: Added a simple API for feature extraction (``torchreid/utils/feature_extractor.py``). See the `documentation <https://kaiyangzhou.github.io/deep-person-reid/user_guide.html>`_ for the instruction.
- [Apr 2020] Code for reproducing the experiments of `deep mutual learning <https://zpascal.net/cvpr2018/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf>`_ in the `OSNet paper <https://arxiv.org/pdf/1905.00953v6.pdf>`__ (Supp. B) has been released at ``projects/DML``.
- [Apr 2020] Upgraded to ``v1.2.0``. The engine class has been made more model-agnostic to improve extensibility. See `Engine <torchreid/engine/engine.py>`_ and `ImageSoftmaxEngine <torchreid/engine/image/softmax.py>`_ for more details. Credit to `Dassl.pytorch <https://github.com/KaiyangZhou/Dassl.pytorch>`_.
- [Dec 2019] Our `OSNet paper <https://arxiv.org/pdf/1905.00953v6.pdf>`_ has been updated, with additional experiments (in section B of the supplementary) showing some useful techniques for improving OSNet's performance in practice.
- [Nov 2019] ``ImageDataManager`` can load training data from target datasets by setting ``load_train_targets=True``, and the train-loader can be accessed with ``train_loader_t = datamanager.train_loader_t``. This feature is useful for domain adaptation research.


Installation
---------------

Make sure `conda <https://www.anaconda.com/distribution/>`_ is installed.


.. code-block:: bash

    # cd to your preferred directory and clone this repo
    git clone https://github.com/KaiyangZhou/deep-person-reid.git

    # create environment
    cd deep-person-reid/
    conda create --name torchreid python=3.7
    conda activate torchreid

    # install dependencies
    # make sure `which python` and `which pip` point to the correct path
    pip install -r requirements.txt

    # install torch and torchvision (select the proper cuda version to suit your machine)
    conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

    # install torchreid (don't need to re-build it if you modify the source code)
    python setup.py develop


Get started: 30 seconds to Torchreid
-------------------------------------
1. Import ``torchreid``

.. code-block:: python
    
    import torchreid

2. Load data manager

.. code-block:: python
    
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources="market1501",
        targets="market1501",
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"]
    )

3 Build model, optimizer and lr_scheduler

.. code-block:: python
    
    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim="adam",
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

4. Build engine

.. code-block:: python
    
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

5. Run training and test

.. code-block:: python
    
    engine.run(
        save_dir="log/resnet50",
        max_epoch=60,
        eval_freq=10,
        print_freq=10,
        test_only=False
    )


A unified interface
-----------------------
In "deep-person-reid/scripts/", we provide a unified interface to train and test a model. See "scripts/main.py" and "scripts/default_config.py" for more details. The folder "configs/" contains some predefined configs which you can use as a starting point.

Below we provide an example to train and test `OSNet (Zhou et al. ICCV'19) <https://arxiv.org/abs/1905.00953>`_. Assume :code:`PATH_TO_DATA` is the directory containing reid datasets. The environmental variable :code:`CUDA_VISIBLE_DEVICES` is omitted, which you need to specify if you have a pool of gpus and want to use a specific set of them.

Conventional setting
^^^^^^^^^^^^^^^^^^^^^

To train OSNet on Market1501, do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    --transforms random_flip random_erase \
    --root $PATH_TO_DATA


The config file sets Market1501 as the default dataset. If you wanna use DukeMTMC-reID, do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    -s dukemtmcreid \
    -t dukemtmcreid \
    --transforms random_flip random_erase \
    --root $PATH_TO_DATA \
    data.save_dir log/osnet_x1_0_dukemtmcreid_softmax_cosinelr

The code will automatically (download and) load the ImageNet pretrained weights. After the training is done, the model will be saved as "log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250". Under the same folder, you can find the `tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ file. To visualize the learning curves using tensorboard, you can run :code:`tensorboard --logdir=log/osnet_x1_0_market1501_softmax_cosinelr` in the terminal and visit :code:`http://localhost:6006/` in your web browser.

Evaluation is automatically performed at the end of training. To run the test again using the trained model, do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
    --root $PATH_TO_DATA \
    model.load_weights log/osnet_x1_0_market1501_softmax_cosinelr/model.pth.tar-250 \
    test.evaluate True


Cross-domain setting
^^^^^^^^^^^^^^^^^^^^^

Suppose you wanna train OSNet on DukeMTMC-reID and test its performance on Market1501, you can do

.. code-block:: bash

    python scripts/main.py \
    --config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml \
    -s dukemtmcreid \
    -t market1501 \
    --transforms random_flip color_jitter \
    --root $PATH_TO_DATA

Here we only test the cross-domain performance. However, if you also want to test the performance on the source dataset, i.e. DukeMTMC-reID, you can set :code:`-t dukemtmcreid market1501`, which will evaluate the model on the two datasets separately.

Different from the same-domain setting, here we replace :code:`random_erase` with :code:`color_jitter`. This can improve the generalization performance on the unseen target dataset.

Pretrained models are available in the `Model Zoo <https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html>`_.


Datasets
--------

Image-reid datasets
^^^^^^^^^^^^^^^^^^^^^
- `Market1501 <https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf>`_
- `CUHK03 <https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_DeepReID_Deep_Filter_2014_CVPR_paper.pdf>`_
- `DukeMTMC-reID <https://arxiv.org/abs/1701.07717>`_
- `MSMT17 <https://arxiv.org/abs/1711.08565>`_
- `VIPeR <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.7285&rep=rep1&type=pdf>`_
- `GRID <http://www.eecs.qmul.ac.uk/~txiang/publications/LoyXiangGong_cvpr_2009.pdf>`_
- `CUHK01 <http://www.ee.cuhk.edu.hk/~xgwang/papers/liZWaccv12.pdf>`_
- `SenseReID <http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Spindle_Net_Person_CVPR_2017_paper.pdf>`_
- `QMUL-iLIDS <http://www.eecs.qmul.ac.uk/~sgg/papers/ZhengGongXiang_BMVC09.pdf>`_
- `PRID <https://pdfs.semanticscholar.org/4c1b/f0592be3e535faf256c95e27982db9b3d3d3.pdf>`_

Geo-localization datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `University-1652 <https://dl.acm.org/doi/abs/10.1145/3394171.3413896>`_

Video-reid datasets
^^^^^^^^^^^^^^^^^^^^^^^
- `MARS <http://www.liangzheng.org/1320.pdf>`_
- `iLIDS-VID <https://www.eecs.qmul.ac.uk/~sgg/papers/WangEtAl_ECCV14.pdf>`_
- `PRID2011 <https://pdfs.semanticscholar.org/4c1b/f0592be3e535faf256c95e27982db9b3d3d3.pdf>`_
- `DukeMTMC-VideoReID <http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Exploit_the_Unknown_CVPR_2018_paper.pdf>`_


Models
-------

ImageNet classification models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- `ResNet <https://arxiv.org/abs/1512.03385>`_
- `ResNeXt <https://arxiv.org/abs/1611.05431>`_
- `SENet <https://arxiv.org/abs/1709.01507>`_
- `DenseNet <https://arxiv.org/abs/1608.06993>`_
- `Inception-ResNet-V2 <https://arxiv.org/abs/1602.07261>`_
- `Inception-V4 <https://arxiv.org/abs/1602.07261>`_
- `Xception <https://arxiv.org/abs/1610.02357>`_
- `IBN-Net <https://arxiv.org/abs/1807.09441>`_

Lightweight models
^^^^^^^^^^^^^^^^^^^
- `NASNet <https://arxiv.org/abs/1707.07012>`_
- `MobileNetV2 <https://arxiv.org/abs/1801.04381>`_
- `ShuffleNet <https://arxiv.org/abs/1707.01083>`_
- `ShuffleNetV2 <https://arxiv.org/abs/1807.11164>`_
- `SqueezeNet <https://arxiv.org/abs/1602.07360>`_

ReID-specific models
^^^^^^^^^^^^^^^^^^^^^^
- `MuDeep <https://arxiv.org/abs/1709.05165>`_
- `ResNet-mid <https://arxiv.org/abs/1711.08106>`_
- `HACNN <https://arxiv.org/abs/1802.08122>`_
- `PCB <https://arxiv.org/abs/1711.09349>`_
- `MLFN <https://arxiv.org/abs/1803.09132>`_
- `OSNet <https://arxiv.org/abs/1905.00953>`_
- `OSNet-AIN <https://arxiv.org/abs/1910.06827>`_


Useful links
-------------
- `OSNet-IBN1-Lite (test-only code with lite docker container) <https://github.com/RodMech/OSNet-IBN1-Lite>`_
- `Deep Learning for Person Re-identification: A Survey and Outlook <https://github.com/mangye16/ReID-Survey>`_


Citation
---------
If you use this code or the models in your research, please give credit to the following papers:

.. code-block:: bash

    @article{torchreid,
      title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch},
      author={Zhou, Kaiyang and Xiang, Tao},
      journal={arXiv preprint arXiv:1910.10093},
      year={2019}
    }
    
    @inproceedings{zhou2019osnet,
      title={Omni-Scale Feature Learning for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      booktitle={ICCV},
      year={2019}
    }

    @article{zhou2021osnet,
      title={Learning Generalisable Omni-Scale Representations for Person Re-Identification},
      author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
      journal={TPAMI},
      year={2021}
    }
