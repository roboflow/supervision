from __future__ import absolute_import, division, print_function

import glob
import os.path as osp

import numpy as np

from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import (
    read_json,
    write_json,
)

from ..dataset import ImageDataset


class VIPeR(ImageDataset):
    """VIPeR.

    Reference:
        Gray et al. Evaluating appearance models for recognition, reacquisition, and tracking. PETS 2007.

    URL: `<https://vision.soe.ucsc.edu/node/178>`_

    Dataset statistics:
        - identities: 632.
        - images: 632 x 2 = 1264.
        - cameras: 2.
    """

    dataset_dir = "viper"
    dataset_url = "http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip"

    def __init__(self, root="", split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.cam_a_dir = osp.join(self.dataset_dir, "VIPeR", "cam_a")
        self.cam_b_dir = osp.join(self.dataset_dir, "VIPeR", "cam_b")
        self.split_path = osp.join(self.dataset_dir, "splits.json")

        required_files = [self.dataset_dir, self.cam_a_dir, self.cam_b_dir]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, "
                "but expected between 0 and {}".format(split_id, len(splits) - 1)
            )
        split = splits[split_id]

        train = split["train"]
        query = split["query"]  # query and gallery share the same images
        gallery = split["gallery"]

        train = [tuple(item) for item in train]
        query = [tuple(item) for item in query]
        gallery = [tuple(item) for item in gallery]

        super(VIPeR, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating 10 random splits of train ids and test ids")

            cam_a_imgs = sorted(glob.glob(osp.join(self.cam_a_dir, "*.bmp")))
            cam_b_imgs = sorted(glob.glob(osp.join(self.cam_b_dir, "*.bmp")))
            assert len(cam_a_imgs) == len(cam_b_imgs)
            num_pids = len(cam_a_imgs)
            print("Number of identities: {}".format(num_pids))
            num_train_pids = num_pids // 2
            """
            In total, there will be 20 splits because each random split creates two
            sub-splits, one using cameraA as query and cameraB as gallery
            while the other using cameraB as query and cameraA as gallery.
            Therefore, results should be averaged over 20 splits (split_id=0~19).

            In practice, a model trained on split_id=0 can be applied to split_id=0&1
            as split_id=0&1 share the same training data (so on and so forth).
            """
            splits = []
            for _ in range(10):
                order = np.arange(num_pids)
                np.random.shuffle(order)
                train_idxs = order[:num_train_pids]
                test_idxs = order[num_train_pids:]
                assert not bool(
                    set(train_idxs) & set(test_idxs)
                ), "Error: train and test overlap"

                train = []
                for pid, idx in enumerate(train_idxs):
                    cam_a_img = cam_a_imgs[idx]
                    cam_b_img = cam_b_imgs[idx]
                    train.append((cam_a_img, pid, 0))
                    train.append((cam_b_img, pid, 1))

                test_a = []
                test_b = []
                for pid, idx in enumerate(test_idxs):
                    cam_a_img = cam_a_imgs[idx]
                    cam_b_img = cam_b_imgs[idx]
                    test_a.append((cam_a_img, pid, 0))
                    test_b.append((cam_b_img, pid, 1))

                # use cameraA as query and cameraB as gallery
                split = {
                    "train": train,
                    "query": test_a,
                    "gallery": test_b,
                    "num_train_pids": num_train_pids,
                    "num_query_pids": num_pids - num_train_pids,
                    "num_gallery_pids": num_pids - num_train_pids,
                }
                splits.append(split)

                # use cameraB as query and cameraA as gallery
                split = {
                    "train": train,
                    "query": test_b,
                    "gallery": test_a,
                    "num_train_pids": num_train_pids,
                    "num_query_pids": num_pids - num_train_pids,
                    "num_gallery_pids": num_pids - num_train_pids,
                }
                splits.append(split)

            print("Totally {} splits are created".format(len(splits)))
            write_json(splits, self.split_path)
            print("Split file saved to {}".format(self.split_path))
