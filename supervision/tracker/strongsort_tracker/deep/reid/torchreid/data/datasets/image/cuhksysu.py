from __future__ import division, print_function, absolute_import
import copy
import glob
import os.path as osp

from ..dataset import ImageDataset


class CUHKSYSU(ImageDataset):
    """CUHKSYSU.

    This dataset can only be used for model training.

    Reference:
        Xiao et al. End-to-end deep learning for person search.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html>`_
    
    Dataset statistics:
        - identities: 11,934
        - images: 34,574
    """
    _train_only = True
    dataset_dir = 'cuhksysu'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.data_dir = osp.join(self.dataset_dir, 'cropped_images')

        # image name format: p11422_s16929_1.jpg
        train = self.process_dir(self.data_dir)
        query = [copy.deepcopy(train[0])]
        gallery = [copy.deepcopy(train[0])]

        super(CUHKSYSU, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dirname):
        img_paths = glob.glob(osp.join(dirname, '*.jpg'))
        # num_imgs = len(img_paths)

        # get all identities:
        pid_container = set()
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = img_name.split('_')[0]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # num_pids = len(pid_container)

        # extract data
        data = []
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = img_name.split('_')[0]
            label = pid2label[pid]
            data.append((img_path, label, 0)) # dummy camera id

        return data
