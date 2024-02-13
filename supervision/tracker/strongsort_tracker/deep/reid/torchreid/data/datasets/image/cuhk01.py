from __future__ import absolute_import, division, print_function

import glob
import os.path as osp
import zipfile

import numpy as np

from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import (
    read_json,
    write_json,
)

from ..dataset import ImageDataset


class CUHK01(ImageDataset):
    """CUHK01.

    Reference:
        Li et al. Human Reidentification with Transferred Metric Learning. ACCV 2012.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html>`_

    Dataset statistics:
        - identities: 971.
        - images: 3884.
        - cameras: 4.

    Note: CUHK01 and CUHK02 overlap.
    """

    dataset_dir = "cuhk01"
    dataset_url = None

    def __init__(self, root="", split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.zip_path = osp.join(self.dataset_dir, "CUHK01.zip")
        self.campus_dir = osp.join(self.dataset_dir, "campus")
        self.split_path = osp.join(self.dataset_dir, "splits.json")

        self.extract_file()

        required_files = [self.dataset_dir, self.campus_dir]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, but expected between 0 and {}".format(
                    split_id, len(splits) - 1
                )
            )
        split = splits[split_id]

        train = split["train"]
        query = split["query"]
        gallery = split["gallery"]

        train = [tuple(item) for item in train]
        query = [tuple(item) for item in query]
        gallery = [tuple(item) for item in gallery]

        super(CUHK01, self).__init__(train, query, gallery, **kwargs)

    def extract_file(self):
        if not osp.exists(self.campus_dir):
            print("Extracting files")
            zip_ref = zipfile.ZipFile(self.zip_path, "r")
            zip_ref.extractall(self.dataset_dir)
            zip_ref.close()

    def prepare_split(self):
        """
        Image name format: 0001001.png, where first four digits represent identity
        and last four digits represent cameras. Camera 1&2 are considered the same
        view and camera 3&4 are considered the same view.
        """
        if not osp.exists(self.split_path):
            print("Creating 10 random splits of train ids and test ids")
            img_paths = sorted(glob.glob(osp.join(self.campus_dir, "*.png")))
            img_list = []
            pid_container = set()
            for img_path in img_paths:
                img_name = osp.basename(img_path)
                pid = int(img_name[:4]) - 1
                camid = (int(img_name[4:7]) - 1) // 2  # result is either 0 or 1
                img_list.append((img_path, pid, camid))
                pid_container.add(pid)

            num_pids = len(pid_container)
            num_train_pids = num_pids // 2

            splits = []
            for _ in range(10):
                order = np.arange(num_pids)
                np.random.shuffle(order)
                train_idxs = order[:num_train_pids]
                train_idxs = np.sort(train_idxs)
                idx2label = {idx: label for label, idx in enumerate(train_idxs)}

                train, test_a, test_b = [], [], []
                for img_path, pid, camid in img_list:
                    if pid in train_idxs:
                        train.append((img_path, idx2label[pid], camid))
                    else:
                        if camid == 0:
                            test_a.append((img_path, pid, camid))
                        else:
                            test_b.append((img_path, pid, camid))

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
