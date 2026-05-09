import os
import json
import numpy as np
import torch
import random
from copy import deepcopy
from marl.config.arguments import get_args
from marl.train.trainer import Trainer
from pprint import pprint

np.set_printoptions(suppress=True, precision=4)

def train(args, return_early=False):
    return Trainer(args).train(return_early=return_early)

if __name__ == '__main__':
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0,10000)
    args.num_updates = args.num_frames // args.num_steps // args.num_processes
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    pprint(vars(args))
    if not args.test:
        with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
            params = deepcopy(vars(args))
            params.pop('device')
            json.dump(params, f)
    train(args)
