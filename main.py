from tools import test_cls
from tools import test_partseg
from tools import test_semseg
from tools import finetune_cls
from tools import finetune_partseg
from tools import finetune_semseg
from tools import svm
from tools import visualiztion
from tools import tsne
from tools import pretrain

from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter

def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)
    # batch size
    if args.vis:
        if args.distributed:
            assert config.total_bs % world_size == 0
            config.dataset.train.others.bs = config.total_bs // world_size
            config.dataset.val.others.bs = 1
            config.dataset.test.others.bs = 1
        else:
            config.dataset.train.others.bs = config.total_bs
            config.dataset.val.others.bs = 1
            config.dataset.test.others.bs = 1
    else:
        if args.distributed:
            assert config.total_bs % world_size == 0
            config.dataset.train.others.bs = config.total_bs // world_size
            config.dataset.val.others.bs = config.total_bs // world_size
            if config.dataset.get('extra_train'):
                config.dataset.extra_train.others.bs = config.total_bs // world_size
            if config.dataset.get('extra_val'):
                config.dataset.extra_val.others.bs = config.total_bs // world_size
        else:
            config.dataset.train.others.bs = config.total_bs
            config.dataset.val.others.bs = config.total_bs
            if config.dataset.get('extra_train'):
                config.dataset.extra_train.others.bs = config.total_bs
            if config.dataset.get('extra_val'):
                config.dataset.extra_val.others.bs = config.total_bs

    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() 

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold

    # run
    if args.test:
        if args.test == 'cls':
            test_cls(args, config)
        elif args.test == 'partseg':
            test_partseg(args, config)
        elif args.test == 'semseg':
            test_semseg(args, config)
    elif args.svm:
        svm(args, config)
    elif args.tsne:
        tsne(args, config)
    elif args.vis:
        visualiztion(args, config)
    else:
        if args.finetune_model or args.scratch_model:
            if args.finetune_model == 'cls' or args.scratch_model == 'cls':
                finetune_cls(args, config, train_writer, val_writer)
            elif args.finetune_model == 'partseg' or args.scratch_model == 'partseg':
                finetune_partseg(args, config, train_writer, val_writer)
            elif args.finetune_model == 'semseg' or args.scratch_model == 'semseg':
                finetune_semseg(args, config, train_writer, val_writer)
        else:
            pretrain(args, config, train_writer, val_writer)

if __name__ == '__main__':
    main()
