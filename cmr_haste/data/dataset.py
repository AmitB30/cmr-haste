import os

import monai as M
from torch.utils.data.dataset import Dataset as TorchDataset
from monai.data.utils import list_data_collate
from monai.transforms import Randomizable
from more_itertools import flatten

from cmr_haste import utils
from cmr_haste.data.utils import partition


def join_paths(d, keys, path):
    for key in keys:
        d[key] = os.path.join(path, d[key])
    return d


def setup_data(
    root,
    partition_data,
    train_split=None,
    val_split=None,
    test_split=None,
    train_transforms=None,
    val_transforms=None,
    test_transforms=None,
    seed=1,
):
    """Setup data for supervised segmentation."""
    metadata = utils.load_json(os.path.join(root, 'metadata.json'))
    data = metadata['data']

    if partition_data:
        data_train, data_val, data_test = partition(
            data,
            seed,
            train_split,
            val_split,
            test_split,
        )
    else:
        data_train = data
        data_val = None
        data_test = None

    if data_train is not None:
        data_train = list(flatten(data_train.values()))
        data_train = ImageSegDataset(data_train, root, transforms=train_transforms)
    if data_val is not None:
        data_val = list(flatten(data_val.values()))
        data_val = ImageSegDataset(data_val, root, transforms=val_transforms)
    if data_test is not None:
        data_test = list(flatten(data_test.values()))
        data_test = ImageSegDataset(data_test, root, transforms=test_transforms)

    return data_train, data_val, data_test


class ImageDataset(TorchDataset, Randomizable):
    """Loads images.

    Args:
        data: List of data dicts. Each dict must contain a path to an image (under the key 'image').
        root: Path to root of dataset.
        transforms: Optional transforms to apply to each data dict.
    """

    def __init__(self, data, root, reader='PILReader', transforms=None):
        self.data = data
        self.root = root
        self.transforms = transforms
        self.reader = M.transforms.LoadImageD(keys=['image'], reader=reader)
        self.set_random_state(seed=M.utils.get_seed())
        self._seed = 0

    def __len__(self):
        return len(self.data)

    def randomize(self, data=None):
        self._seed = self.R.randint(M.utils.MAX_SEED, dtype='uint32')

    def _transform(self, data):
        if isinstance(self.transforms, Randomizable):
            self.transforms.set_random_state(seed=self._seed)
        data = M.transforms.apply_transform(self.transforms, data)
        return data

    def _read(self, data_dict):
        data_dict = join_paths(data_dict, ['image'], self.root)
        data_dict = self.reader(data_dict)
        return data_dict

    def __getitem__(self, index):
        self.randomize()
        data_dict = self.data[index]
        data_dict = self._read(data_dict)
        if self.transforms is not None:
            data_dict = self._transform(data_dict)
        return data_dict

    def get_study(self, name):
        data_dicts = [i for i in self.data if i['meta']['study_name'] == name]
        if not data_dicts:
            raise ValueError(f'Study `{name}` not found')
        data_dicts = [self._read(i) for i in data_dicts]
        return data_dicts

    def load_study(self, name, collate=True):
        data_dicts = self.get_study(name)
        if self.transforms is not None:
            data_dicts = self._transform(data_dicts)
        if collate:
            data_dicts = list_data_collate(data_dicts)
        return data_dicts

    def iter_studies(self):
        data = utils.groupby(self.data, key=lambda x: x['meta']['study_name'])
        for name, data_dicts in data.items():
            data_dicts = [self._read(i) for i in data_dicts]
            if self.transforms is not None:
                data_dicts = self._transform(data_dicts)
            data_dicts = list_data_collate(data_dicts)
            yield name, data_dicts

    def study_names(self):
        names = [i['meta']['study_name'] for i in self.data]
        names = list(set(names))
        return names

    def study_uids(self):
        uids = [i['meta']['study_uid'] for i in self.data]
        uids = list(set(uids))
        return uids


class ImageSegDataset(TorchDataset, Randomizable):
    """Loads images and segmentations.

    Args:
        data: A list of data dicts. Each dict must contain paths to an image (under the key 'image')
            and a segmentation (under the key 'seg').
        root: A path to the root of the dataset.
        transforms: Optional transforms to apply to each data dict.
    """

    def __init__(self, data, root, reader='PILReader', transforms=None):
        self.data = data
        self.root = root
        self.transforms = transforms
        self.reader = M.transforms.LoadImageD(keys=['image', 'seg'], reader=reader)
        self.image_key = 'image'
        self.seg_key = 'seg'
        self.keys = [self.image_key, self.seg_key]
        self.set_random_state(seed=M.utils.get_seed())
        self._seed = 0

    def __len__(self):
        return len(self.data)

    def randomize(self, data=None):
        self._seed = self.R.randint(M.utils.MAX_SEED, dtype='uint32')

    def _transform(self, data):
        if isinstance(self.transforms, Randomizable):
            self.transforms.set_random_state(seed=self._seed)
        data = M.transforms.apply_transform(self.transforms, data)
        return data

    def _read(self, data_dict):
        data_dict = join_paths(data_dict, self.keys, self.root)
        data_dict = self.reader(data_dict)
        return data_dict

    def __getitem__(self, index):
        self.randomize()
        data_dict = self.data[index]
        data_dict = self._read(data_dict)
        if self.transforms is not None:
            data_dict = self._transform(data_dict)
        return data_dict

    def get_study(self, name):
        data_dicts = [i for i in self.data if i['meta']['study_name'] == name]
        if not data_dicts:
            raise ValueError(f'Study `{name}` not found')
        data_dicts = [self._read(i) for i in data_dicts]
        return data_dicts

    def load_study(self, name, collate=True):
        data_dicts = self.get_study(name)
        if self.transforms is not None:
            data_dicts = self._transform(data_dicts)
        if collate:
            data_dicts = list_data_collate(data_dicts)
        return data_dicts

    def iter_studies(self):
        data = utils.groupby(self.data, key=lambda x: x['meta']['study_name'])
        for name, data_dicts in data.items():
            data_dicts = [self._read(i) for i in data_dicts]
            if self.transforms is not None:
                data_dicts = self._transform(data_dicts)
            data_dicts = list_data_collate(data_dicts)
            yield name, data_dicts

    def study_names(self):
        names = [i['meta']['study_name'] for i in self.data]
        names = list(set(names))
        return names

    def study_uids(self):
        uids = [i['meta']['study_uid'] for i in self.data]
        uids = list(set(uids))
        return uids
