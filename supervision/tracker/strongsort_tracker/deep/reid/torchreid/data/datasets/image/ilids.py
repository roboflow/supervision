from __future__ import division, print_function, absolute_import
import copy
import glob
import random
import os.path as osp
from collections import defaultdict

from supervision.tracker.strongsort_tracker.deep.reid.torchreid.utils import read_json, write_json

from ..dataset import ImageDataset


class iLIDS(ImageDataset):
    """QMUL-iLIDS.

    Reference:
        Zheng et al. Associating Groups of People. BMVC 2009.
    
    Dataset statistics:
        - identities: 119.
        - images: 476.
        - cameras: 8 (not explicitly provided).
    """
    dataset_dir = 'ilids'
    dataset_url = 'http://www.eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz'

    def __init__(self, root='', split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.data_dir = osp.join(self.dataset_dir, 'i-LIDS_Pedestrian/Persons')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')

        required_files = [self.dataset_dir, self.data_dir]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but '
                'expected between 0 and {}'.format(split_id,
                                                   len(splits) - 1)
            )
        split = splits[split_id]

        train, query, gallery = self.process_split(split)

        super(iLIDS, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating splits ...')

            paths = glob.glob(osp.join(self.data_dir, '*.jpg'))
            img_names = [osp.basename(path) for path in paths]
            num_imgs = len(img_names)
            assert num_imgs == 476, 'There should be 476 images, but ' \
                                    'got {}, please check the data'.format(num_imgs)

            # store image names
            # image naming format:
            #   the first four digits denote the person ID
            #   the last four digits denote the sequence index
            pid_dict = defaultdict(list)
            for img_name in img_names:
                pid = int(img_name[:4])
                pid_dict[pid].append(img_name)
            pids = list(pid_dict.keys())
            num_pids = len(pids)
            assert num_pids == 119, 'There should be 119 identities, ' \
                                    'but got {}, please check the data'.format(num_pids)

            num_train_pids = int(num_pids * 0.5)

            splits = []
            for _ in range(10):
                # randomly choose num_train_pids train IDs and the rest for test IDs
                pids_copy = copy.deepcopy(pids)
                random.shuffle(pids_copy)
                train_pids = pids_copy[:num_train_pids]
                test_pids = pids_copy[num_train_pids:]

                train = []
                query = []
                gallery = []

                # for train IDs, all images are used in the train set.
                for pid in train_pids:
                    img_names = pid_dict[pid]
                    train.extend(img_names)

                # for each test ID, randomly choose two images, one for
                # query and the other one for gallery.
                for pid in test_pids:
                    img_names = pid_dict[pid]
                    samples = random.sample(img_names, 2)
                    query.append(samples[0])
                    gallery.append(samples[1])

                split = {'train': train, 'query': query, 'gallery': gallery}
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file is saved to {}'.format(self.split_path))

    def get_pid2label(self, img_names):
        pid_container = set()
        for img_name in img_names:
            pid = int(img_name[:4])
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def parse_img_names(self, img_names, pid2label=None):
        data = []

        for img_name in img_names:
            pid = int(img_name[:4])
            if pid2label is not None:
                pid = pid2label[pid]
            camid = int(img_name[4:7]) - 1 # 0-based
            img_path = osp.join(self.data_dir, img_name)
            data.append((img_path, pid, camid))

        return data

    def process_split(self, split):
        train_pid2label = self.get_pid2label(split['train'])
        train = self.parse_img_names(split['train'], train_pid2label)
        query = self.parse_img_names(split['query'])
        gallery = self.parse_img_names(split['gallery'])
        return train, query, gallery
