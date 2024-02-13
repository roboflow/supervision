from __future__ import absolute_import, division, print_function

import os.path as osp
import random

from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import (
    read_json,
    write_json,
)

from ..dataset import ImageDataset


class PRID(ImageDataset):
    """PRID (single-shot version of prid-2011)

    Reference:
        Hirzer et al. Person Re-Identification by Descriptive and Discriminative
        Classification. SCIA 2011.

    URL: `<https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/>`_

    Dataset statistics:
        - Two views.
        - View A captures 385 identities.
        - View B captures 749 identities.
        - 200 identities appear in both views (index starts from 1 to 200).
    """

    dataset_dir = "prid2011"
    dataset_url = None
    _junk_pids = list(range(201, 750))

    def __init__(self, root="", split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.cam_a_dir = osp.join(self.dataset_dir, "prid_2011", "single_shot", "cam_a")
        self.cam_b_dir = osp.join(self.dataset_dir, "prid_2011", "single_shot", "cam_b")
        self.split_path = osp.join(self.dataset_dir, "splits_single_shot.json")

        required_files = [self.dataset_dir, self.cam_a_dir, self.cam_b_dir]
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

        train, query, gallery = self.process_split(split)

        super(PRID, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits ...")

            splits = []
            for _ in range(10):
                # randomly sample 100 IDs for train and use the rest 100 IDs for test
                # (note: there are only 200 IDs appearing in both views)
                pids = [i for i in range(1, 201)]
                train_pids = random.sample(pids, 100)
                train_pids.sort()
                test_pids = [i for i in pids if i not in train_pids]
                split = {"train": train_pids, "test": test_pids}
                splits.append(split)

            print("Totally {} splits are created".format(len(splits)))
            write_json(splits, self.split_path)
            print("Split file is saved to {}".format(self.split_path))

    def process_split(self, split):
        train_pids = split["train"]
        test_pids = split["test"]

        train_pid2label = {pid: label for label, pid in enumerate(train_pids)}

        # train
        train = []
        for pid in train_pids:
            img_name = "person_" + str(pid).zfill(4) + ".png"
            pid = train_pid2label[pid]
            img_a_path = osp.join(self.cam_a_dir, img_name)
            train.append((img_a_path, pid, 0))
            img_b_path = osp.join(self.cam_b_dir, img_name)
            train.append((img_b_path, pid, 1))

        # query and gallery
        query, gallery = [], []
        for pid in test_pids:
            img_name = "person_" + str(pid).zfill(4) + ".png"
            img_a_path = osp.join(self.cam_a_dir, img_name)
            query.append((img_a_path, pid, 0))
            img_b_path = osp.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))
        for pid in range(201, 750):
            img_name = "person_" + str(pid).zfill(4) + ".png"
            img_b_path = osp.join(self.cam_b_dir, img_name)
            gallery.append((img_b_path, pid, 1))

        return train, query, gallery
