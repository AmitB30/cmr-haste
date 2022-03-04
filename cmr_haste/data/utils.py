import numpy as np
from monai.data import partition_dataset


def partition(data, seed, train_split, val_split=None, test_split=None):
    """Partition data into train, val, and test sets.

    Args:
        data: Dict mapping study name to cine data dicts.
        seed: Seed used for random shuffling.
        train_split: Ratio of training data.
        val_split: Ratio of validation data.
        test_split: Ratio of test data.

    Returns:
        Training, validation, and test sets, of which the latter two may be None.
    """
    ratios = [train_split, val_split, test_split]
    total = np.sum([i for i in ratios if i is not None])
    if total != 1.0:
        raise ValueError(f'Partition ratios should sum to 1.0, found {total}')

    data = sorted(list(data.items()), key=lambda x: x[0])
    if val_split and test_split:
        partitions = partition_dataset(data, ratios=ratios, shuffle=True, seed=seed)
        data_train, data_val, data_test = [dict(i) for i in partitions]
    elif val_split and not test_split:
        partitions = partition_dataset(data, ratios=[ratios[0], ratios[1]], shuffle=True, seed=seed)
        data_train, data_val = [dict(i) for i in partitions]
        data_test = None
    elif not val_split and test_split:
        partitions = partition_dataset(data, ratios=[ratios[0], ratios[2]], shuffle=True, seed=seed)
        data_train, data_test = [dict(i) for i in partitions]
        data_val = None
    else:
        data_train = data
        data_val, data_test = None, None

    return data_train, data_val, data_test