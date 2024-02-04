from __future__ import division, print_function, absolute_import
import os.path as osp
import warnings
from scipy.io import loadmat

from ..dataset import VideoDataset


class Mars(VideoDataset):
    """MARS.

    Reference:
        Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

    URL: `<http://www.liangzheng.com.cn/Project/project_mars.html>`_
    
    Dataset statistics:
        - identities: 1261.
        - tracklets: 8298 (train) + 1980 (query) + 9330 (gallery).
        - cameras: 6.
    """
    dataset_dir = 'mars'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.train_name_path = osp.join(
            self.dataset_dir, 'info/train_name.txt'
        )
        self.test_name_path = osp.join(self.dataset_dir, 'info/test_name.txt')
        self.track_train_info_path = osp.join(
            self.dataset_dir, 'info/tracks_train_info.mat'
        )
        self.track_test_info_path = osp.join(
            self.dataset_dir, 'info/tracks_test_info.mat'
        )
        self.query_IDX_path = osp.join(self.dataset_dir, 'info/query_IDX.mat')

        required_files = [
            self.dataset_dir, self.train_name_path, self.test_name_path,
            self.track_train_info_path, self.track_test_info_path,
            self.query_IDX_path
        ]
        self.check_before_run(required_files)

        train_names = self.get_names(self.train_name_path)
        test_names = self.get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path
                              )['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path
                             )['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path
                            )['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX, :]
        gallery_IDX = [
            i for i in range(track_test.shape[0]) if i not in query_IDX
        ]
        track_gallery = track_test[gallery_IDX, :]

        train = self.process_data(
            train_names, track_train, home_dir='bbox_train', relabel=True
        )
        query = self.process_data(
            test_names, track_query, home_dir='bbox_test', relabel=False
        )
        gallery = self.process_data(
            test_names, track_gallery, home_dir='bbox_test', relabel=False
        )

        super(Mars, self).__init__(train, query, gallery, **kwargs)

    def get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def process_data(
        self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0
    ):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))

        if relabel:
            pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1:
                continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel:
                pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(
                set(pnames)
            ) == 1, 'Error: a single tracklet contains different person images'

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(
                set(camnames)
            ) == 1, 'Error: images are captured under different cameras!'

            # append image names with directory information
            img_paths = [
                osp.join(self.dataset_dir, home_dir, img_name[:4], img_name)
                for img_name in img_names
            ]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))

        return tracklets

    def combine_all(self):
        warnings.warn(
            'Some query IDs do not appear in gallery. Therefore, combineall '
            'does not make any difference to Mars'
        )
