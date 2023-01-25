import logging
import wandb
import os
import torch
import glob
import json
from attrdict import AttrDict
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s][%(process)05d] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white,bold',
        'INFOV': 'cyan,bold',
        'WARNING': 'yellow',
        'ERROR': 'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

fh = logging.FileHandler('log.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

log = logging.getLogger('rl')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)
log.addHandler(fh)


def set_file_handler(logdir):
    global fh
    global log
    log.removeHandler(fh)
    filepath = os.path.join(logdir, 'log.txt')
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def config_wandb(**kwargs):
    # wandb initialization
    wandb.init(project=kwargs['wandb_project'], entity='qdrl', group=kwargs['wandb_group'], name=kwargs['run_name'])
    cfg = kwargs.get('cfg', None)
    if cfg is None:
        cfg = {}
        for key, val in kwargs.items():
            cfg[key] = val
    wandb.config.update(cfg)


def save_checkpoint(cp_dir, cp_name, model, optimizer, **kwargs):
    os.makedirs(cp_dir, exist_ok=True)
    params = {}
    params['model_state_dict'] = model.state_dict()
    params['optim_state_dict'] = optimizer.state_dict()
    for key, val in kwargs:
        params[key] = val
    torch.save(params, os.path.join(cp_dir, cp_name))


def save_cfg(dir, cfg):
    def to_dict(cfg):
        if isinstance(cfg, AttrDict):
            cfg = dict(cfg)
    filename = 'cfg.json'
    fp = os.path.join(dir, filename)
    with open(fp, 'w') as f:
        json.dump(cfg, f, default=to_dict, indent=4)


def get_checkpoints(checkpoints_dir):
    checkpoints = glob.glob(os.path.join(checkpoints_dir, 'cp_*'))
    return sorted(checkpoints)
