.. _datasets:

Datasets
=========

Here we provide a comprehensive guide on how to prepare the datasets.

Suppose you want to store the reid data in a directory called "path/to/reid-data/", you need to specify the ``root`` as *root='path/to/reid-data/'* when initializing ``DataManager``. Below we use ``$REID`` to denote "path/to/reid-data".

Please refer to :ref:`torchreid_data` for details regarding the arguments.


.. note::
    Dataset with a :math:`\dagger` symbol means that the process is automated, so you can directly call the dataset in ``DataManager`` (which automatically downloads the dataset and organizes the data structure). However, we also provide a way below to help the manual setup in case the automation fails.


.. note::
    The keys to use specific datasets are enclosed in the parantheses beside the datasets' names.


.. note::
    You are suggested to use the provided names for dataset folders such as "market1501" for Market1501 and "dukemtmcreid" for DukeMTMC-reID when doing the manual setup, otherwise you need to modify the source code accordingly (i.e. the ``dataset_dir`` attribute).

.. note::
    Some download links provided by the original authors might not work. You can email `Kaiyang Zhou <https://kaiyangzhou.github.io/>`_ to reqeust new links. Please do provide your full name, institution, and purpose of using the data in the email (best use your work email address).

.. contents::
   :local:


Image Datasets
--------------

Market1501 :math:`^\dagger` (``market1501``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Create a directory named "market1501" under ``$REID``.
- Download the dataset to "market1501" from http://www.liangzheng.org/Project/project_reid.html and extract the files.
- The data structure should look like

.. code-block:: none
    
    market1501/
        Market-1501-v15.09.15/
            query/
            bounding_box_train/
            bounding_box_test/

- To use the extra 500K distractors (i.e. Market1501 + 500K), go to the **Market-1501+500k Dataset** section at http://www.liangzheng.org/Project/project_reid.html, download the zip file "distractors_500k.zip" and extract it under "market1501/Market-1501-v15.09.15". The argument to use these 500K distrctors is ``market1501_500k`` in ``ImageDataManager``.


CUHK03 (``cuhk03``)
^^^^^^^^^^^^^^^^^^^^^
- Create a folder named "cuhk03" under ``$REID``.
- Download the dataset to "cuhk03/" from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html and extract "cuhk03_release.zip", resulting in "cuhk03/cuhk03_release/".
- Download the new split (767/700) from `person-re-ranking <https://github.com/zhunzhong07/person-re-ranking/tree/master/evaluation/data/CUHK03>`_. What you need are "cuhk03_new_protocol_config_detected.mat" and "cuhk03_new_protocol_config_labeled.mat". Put these two mat files under "cuhk03/".
- The data structure should look like

.. code-block:: none
    
    cuhk03/
        cuhk03_release/
        cuhk03_new_protocol_config_detected.mat
        cuhk03_new_protocol_config_labeled.mat


- In the default mode, we load data using the new split (767/700). If you wanna use the original (20) splits (1367/100), please set ``cuhk03_classic_split`` to True in ``ImageDataManager``. As the CMC is computed differently from Market1501 for the 1367/100 split (see `here <http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html>`_), you need to enable ``use_metric_cuhk03`` in ``ImageDataManager`` to activate the *single-gallery-shot* metric for fair comparison with some methods that adopt the old splits (*do not need to report mAP*). In addition, we support both *labeled* and *detected* modes. The default mode loads *detected* images. Enable ``cuhk03_labeled`` in ``ImageDataManager`` if you wanna train and test on *labeled* images.

.. note::
    The code will extract images in "cuhk-03.mat" and save them under "cuhk03/images_detected" and "cuhk03/images_labeled". Also, four json files will be automatically generated, i.e. "splits_classic_detected.json", "splits_classic_labeled.json", "splits_new_detected.json" and "splits_new_labeled.json". If the parent path of ``$REID`` is changed, these json files should be manually deleted. The code can automatically generate new json files to match the new path.
    

DukeMTMC-reID :math:`^\dagger` (``dukemtmcreid``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Create a directory called "dukemtmc-reid" under ``$REID``.
- Download "DukeMTMC-reID" from http://vision.cs.duke.edu/DukeMTMC/ and extract it under "dukemtmc-reid".
- The data structure should look like

.. code-block:: none
    
    dukemtmc-reid/
        DukeMTMC-reID/
            query/
            bounding_box_train/
            bounding_box_test/
            ...

MSMT17 (``msmt17``)
^^^^^^^^^^^^^^^^^^^^^
- Create a directory called "msmt17" under ``$REID``.
- Download the dataset from http://www.pkuvmc.com/publications/msmt17.html to "msmt17" and extract the files.
- The data structure should look like

.. code-block:: none
    
    msmt17/
        MSMT17_V1/ # or MSMT17_V2
            train/
            test/
            list_train.txt
            list_query.txt
            list_gallery.txt
            list_val.txt

VIPeR :math:`^\dagger` (``viper``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The download link is http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip.
- Organize the dataset in a folder named "viper" as follows

.. code-block:: none
    
    viper/
        VIPeR/
            cam_a/
            cam_b/

GRID :math:`^\dagger` (``grid``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The download link is http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip.
- Organize the dataset in a folder named "grid" as follows

.. code-block:: none
    
    grid/
        underground_reid/
            probe/
            gallery/
            ...

CUHK01 (``cuhk01``)
^^^^^^^^^^^^^^^^^^^^^^^^
- Create a folder named "cuhk01" under ``$REID``.
- Download "CUHK01.zip" from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html and place it under "cuhk01/".
- The code can automatically extract the files, or you can do it yourself.
- The data structure should look like

.. code-block:: none
    
    cuhk01/
        campus/

SenseReID (``sensereid``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Create "sensereid" under ``$REID``.
- Download the dataset from this `link <https://drive.google.com/file/d/0B56OfSrVI8hubVJLTzkwV2VaOWM/view>`_ and extract it to "sensereid".
- Organize the data to be like

.. code-block:: none

    sensereid/
        SenseReID/
            test_probe/
            test_gallery/

QMUL-iLIDS :math:`^\dagger` (``ilids``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Create a folder named "ilids" under ``$REID``.
- Download the dataset from http://www.eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz and organize it to look like

.. code-block:: none
    
    ilids/
        i-LIDS_Pedestrian/
            Persons/

PRID (``prid``)
^^^^^^^^^^^^^^^^^^^
- Create a directory named "prid2011" under ``$REID``.
- Download the dataset from https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/ and extract it under "prid2011".
- The data structure should end up with

.. code-block:: none

    prid2011/
        prid_2011/
            single_shot/
            multi_shot/

CUHK02 (``cuhk02``)
^^^^^^^^^^^^^^^^^^^^^
- Create a folder named "cuhk02" under ``$REID``.
- Download the data from http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html and put it under "cuhk02/".
- Extract the file so the data structure looks like

.. code-block:: none
    
    cuhk02/
        Dataset/
            P1/
            P2/
            P3/
            P4/
            P5/

CUHKSYSU (``cuhksysu``)
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Create a folder named "cuhksysu" under ``$REID``.
- Download the data to "cuhksysu/" from this `google drive link <https://drive.google.com/file/d/1XmiNVrfK2ZmI0ZZ2HHT80HHbDrnE4l3W/view?usp=sharing>`_.
- Extract the zip file under "cuhksysu/".
- The data structure should look like

.. code-block:: none
    
    cuhksysu/
        cropped_images


Video Datasets
--------------

MARS (``mars``)
^^^^^^^^^^^^^^^^^
- Create "mars/" under ``$REID``.
- Download the dataset from http://www.liangzheng.com.cn/Project/project_mars.html and place it in "mars/".
- Extract "bbox_train.zip" and "bbox_test.zip".
- Download the split metadata from https://github.com/liangzheng06/MARS-evaluation/tree/master/info and put "info/" in "mars/".
- The data structure should end up with

.. code-block:: none
    
    mars/
        bbox_test/
        bbox_train/
        info/

iLIDS-VID :math:`^\dagger` (``ilidsvid``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Create "ilids-vid" under ``$REID``.
- Download the dataset from https://xiatian-zhu.github.io/downloads_qmul_iLIDS-VID_ReID_dataset.html to "ilids-vid".
- Organize the data structure to match

.. code-block:: none
    
    ilids-vid/
        i-LIDS-VID/
        train-test people splits/

PRID2011 (``prid2011``)
^^^^^^^^^^^^^^^^^^^^^^^^^
- Create a directory named "prid2011" under ``$REID``.
- Download the dataset from https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/ and extract it under "prid2011".
- Download the split created by *iLIDS-VID* from `this google drive <https://drive.google.com/open?id=1qw7SI7YdIgfHetIQO7LLW4SHpL_qkieT>`_ and put it under "prid2011/". Following the standard protocol, only 178 persons whose sequences are more than a threshold are used.
- The data structure should end up with

.. code-block:: none

    prid2011/
        splits_prid2011.json
        prid_2011/
            single_shot/
            multi_shot/

DukeMTMC-VideoReID :math:`^\dagger` (``dukemtmcvidreid``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Create "dukemtmc-vidreid" under ``$REID``.
- Download "DukeMTMC-VideoReID" from http://vision.cs.duke.edu/DukeMTMC/ and unzip the file to "dukemtmc-vidreid/".
- The data structure should look like

.. code-block:: none
    
    dukemtmc-vidreid/
        DukeMTMC-VideoReID/
            train/
            query/
            gallery/
