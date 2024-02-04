from __future__ import division, print_function, absolute_import
import copy
import glob
import os.path as osp

from ..dataset import ImageDataset


class SenseReID(ImageDataset):
    """SenseReID.

    This dataset is used for test purpose only.

    Reference:
        Zhao et al. Spindle Net: Person Re-identification with Human Body
        Region Guided Feature Decomposition and Fusion. CVPR 2017.

    URL: `<https://drive.google.com/file/d/0B56OfSrVI8hubVJLTzkwV2VaOWM/view>`_

    Dataset statistics:
        - query: 522 ids, 1040 images.
        - gallery: 1717 ids, 3388 images.
    """
    dataset_dir = 'sensereid'
    dataset_url = None

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.query_dir = osp.join(self.dataset_dir, 'SenseReID', 'test_probe')
        self.gallery_dir = osp.join(
            self.dataset_dir, 'SenseReID', 'test_gallery'
        )

        required_files = [self.dataset_dir, self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)

        query = self.process_dir(self.query_dir)
        gallery = self.process_dir(self.gallery_dir)

        # relabel
        g_pids = set()
        for _, pid, _ in gallery:
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        query = [
            (img_path, pid2label[pid], camid) for img_path, pid, camid in query
        ]
        gallery = [
            (img_path, pid2label[pid], camid)
            for img_path, pid, camid in gallery
        ]
        train = copy.deepcopy(query) + copy.deepcopy(gallery) # dummy variable

        super(SenseReID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        data = []

        for img_path in img_paths:
            img_name = osp.splitext(osp.basename(img_path))[0]
            pid, camid = img_name.split('_')
            pid, camid = int(pid), int(camid)
            data.append((img_path, pid, camid))

        return data
