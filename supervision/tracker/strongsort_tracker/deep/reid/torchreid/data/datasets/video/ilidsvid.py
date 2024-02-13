from __future__ import absolute_import, division, print_function

import glob
import os.path as osp

from scipy.io import loadmat

from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import (
    read_json,
    write_json,
)

from ..dataset import VideoDataset


class iLIDSVID(VideoDataset):
    """iLIDS-VID.

    Reference:
        Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.

    URL: `<http://www.eecs.qmul.ac.uk/~xiatian/downloads_qmul_iLIDS-VID_ReID_dataset.html>`_

    Dataset statistics:
        - identities: 300.
        - tracklets: 600.
        - cameras: 2.
    """

    dataset_dir = "ilids-vid"
    dataset_url = "http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar"

    def __init__(self, root="", split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.data_dir = osp.join(self.dataset_dir, "i-LIDS-VID")
        self.split_dir = osp.join(self.dataset_dir, "train-test people splits")
        self.split_mat_path = osp.join(self.split_dir, "train_test_splits_ilidsvid.mat")
        self.split_path = osp.join(self.dataset_dir, "splits.json")
        self.cam_1_path = osp.join(self.dataset_dir, "i-LIDS-VID/sequences/cam1")
        self.cam_2_path = osp.join(self.dataset_dir, "i-LIDS-VID/sequences/cam2")

        required_files = [self.dataset_dir, self.data_dir, self.split_dir]
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
        train_dirs, test_dirs = split["train"], split["test"]

        train = self.process_data(train_dirs, cam1=True, cam2=True)
        query = self.process_data(test_dirs, cam1=True, cam2=False)
        gallery = self.process_data(test_dirs, cam1=False, cam2=True)

        super(iLIDSVID, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits ...")
            mat_split_data = loadmat(self.split_mat_path)["ls_set"]

            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids // 2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = sorted(glob.glob(osp.join(self.cam_1_path, "*")))
            person_cam2_dirs = sorted(glob.glob(osp.join(self.cam_2_path, "*")))

            person_cam1_dirs = [osp.basename(item) for item in person_cam1_dirs]
            person_cam2_dirs = [osp.basename(item) for item in person_cam2_dirs]

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split, num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split, :num_ids_each]))

                train_idxs = [int(i) - 1 for i in train_idxs]
                test_idxs = [int(i) - 1 for i in test_idxs]

                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]

                split = {"train": train_dirs, "test": test_dirs}
                splits.append(split)

            print(
                "Totally {} splits are created, following Wang et al. ECCV'14".format(
                    len(splits)
                )
            )
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

    def process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, "*.png"))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))

            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, "*.png"))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))

        return tracklets
