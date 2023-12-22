import pdb
from .prepare import prepare_model, prepare_train, init, init_gs
from .train import train
from .train_gs import train_gs
from .evaluate import evaluate
import torch
import torch.nn as nn
from functools import partial


class Trainer:
    def __init__(self, args, config):
        self.config = config
        self.args = args
        self.flags = {}
        self.split_info = None

        self.prepare_train = partial(prepare_train, self)
        self.prepare_model = partial(prepare_model, self)
        self.init          = partial(init, self)
        self.init_gs       = partial(init_gs, self)

        self.train_gs      = partial(train_gs, self)
        self.train         = partial(train, self)
        self.evaluation    = partial(evaluate, self)