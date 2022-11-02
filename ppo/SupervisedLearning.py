import torch
import numpy as np
import nmmo
from nmmo import config
from neurips2022nmmo import CompetitionConfig, TeamBasedEnv
from nmmo.io import action

import yaml
from easydict import EasyDict

from model.Arch import Archtecture

class SupervisedLearning:
    def __init__(self, args):
        self.arch = Archtecture(args)
        self.args = args

if __name__ == '__main__':
    with open("./scripts/config/config.yml", 'r') as f:
        arg_file = yaml.full_load(f)
    args = EasyDict(arg_file)
    sl = SupervisedLearning(args)
