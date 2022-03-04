"""Miscellaneous utilities."""
import itertools
import json
from pathlib import Path


def ensure_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)


def ensure_dirs(paths):
    for path in paths:
        ensure_dir(path)


def load_json(path):
    with open(path) as f:
        json_data = json.load(f)
    return json_data


def groupby(iterable, key=None):
    """Sort and group iterable by key."""
    groups = []
    keys = []
    iterable = sorted(iterable, key=key)
    for k, g in itertools.groupby(iterable, key):
        groups.append(list(g))
        keys.append(k)
    return {k: v for k, v in zip(keys, groups)}

