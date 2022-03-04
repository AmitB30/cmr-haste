"""Output and summary utilities.

Adapted from timm.
"""
import csv
import datetime
import os
from collections import OrderedDict

from cmr_haste.utils.misc import ensure_dir, ensure_dirs

try:
    import wandb
except ImportError:
    pass


def get_out_dir(root):
    """Get output directory."""
    d = datetime.datetime.today().strftime('%y-%m-%d')
    t = datetime.datetime.today().strftime('%H-%M-%S')
    path = os.path.join(root, d, t)
    ensure_dir(path)
    return path


def get_out_dirs(root, exp_name):
    """Get output directories."""
    output_dir = os.path.join(root, exp_name)
    logs_dir = os.path.join(output_dir, 'logs')
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    recovery_dir = os.path.join(output_dir, 'recovery')
    ensure_dirs([logs_dir, checkpoint_dir, recovery_dir])
    return output_dir, logs_dir, checkpoint_dir, recovery_dir


def update_summary(epoch, train_metrics, filename, val_metrics=None, write_header=False,
                   writer=None, log_wandb=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if val_metrics is not None:
        rowd.update([('val_' + k, v) for k, v in val_metrics.items()])
    if writer is not None:
        for k, v in rowd.items():
            writer.add_scalar(k, v, epoch)
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as f:
        dw = csv.DictWriter(f, fieldnames=rowd.keys())
        if write_header:
            dw.writeheader()
        dw.writerow(rowd)
