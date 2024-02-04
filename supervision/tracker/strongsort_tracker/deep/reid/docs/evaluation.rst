Evaluation
==========

Image ReID
-----------
- **Market1501**, **DukeMTMC-reID**, **CUHK03 (767/700 split)** and **MSMT17** have fixed split so keeping ``split_id=0`` is fine.
- **CUHK03 (classic split)** has 20 fixed splits, so do ``split_id=0~19``.
- **VIPeR** contains 632 identities each with 2 images under two camera views. Evaluation should be done for 10 random splits. Each split randomly divides 632 identities to 316 train ids (632 images) and the other 316 test ids (632 images). Note that, in each random split, there are two sub-splits, one using camera-A as query and camera-B as gallery while the other one using camera-B as query and camera-A as gallery. Thus, there are totally 20 splits generated with ``split_id`` starting from 0 to 19. Models can be trained on ``split_id=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]`` (because ``split_id=0`` and ``split_id=1`` share the same train set, and so on and so forth.). At test time, models trained on ``split_id=0`` can be directly evaluated on ``split_id=1``, models trained on ``split_id=2`` can be directly evaluated on ``split_id=3``, and so on and so forth.
- **CUHK01** is similar to VIPeR in the split generation.
- **GRID** , **iLIDS** and **PRID** have 10 random splits, so evaluation should be done by varying ``split_id`` from 0 to 9.
- **SenseReID** has no training images and is used for evaluation only.


.. note::
    The ``split_id`` argument is defined in ``ImageDataManager`` and ``VideoDataManager``. Please refer to :ref:`torchreid_data`.


Video ReID
-----------
- **MARS** and **DukeMTMC-VideoReID** have fixed single split so using ``split_id=0`` is ok.
- **iLIDS-VID** and **PRID2011** have 10 predefined splits so evaluation should be done by varying ``split_id`` from 0 to 9.