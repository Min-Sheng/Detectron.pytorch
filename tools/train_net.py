""" Training Script """

import argparse
#import distutils.util
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb import combined_roidb
from modeling.model_builder import Generalized_RCNN
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import log_stats
from utils.timer import Timer
from utils.training_stats import TrainingStats

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--dataset', dest='dataset', required=True,
        help='Dataset to use')
    parser.add_argument('--g', dest='group',
                      help='which group to train',
                      default=1, type=int)
    parser.add_argument('--seen', dest='seen',default=1, type=int)
    #parser.add_argument(
    #    '--cfg', dest='cfg_file', required=True,
    #    help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=10, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_epochs',
        help='Epochs to decay the learning rate on. '
             'Decay happens on the beginning of a epoch. '
             'Epoch is 0-indexed.',
        default=[10, 20], nargs='+', type=int)

    # Epoch
    parser.add_argument(
        '--start_iter',
        help='Starting iteration for first training epoch. 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--start_epoch',
        help='Starting epoch count. Epoch is 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--epochs', dest='num_epochs',
        help='Number of epochs to train',
        default=30, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--ckpt_num_per_epoch',
        help='number of checkpoints to save in each epoch. '
             'Not include the one at the end of an epoch.',
        default=1, type=int)

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    return parser.parse_args()


def main():
    """Main function"""

    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    if args.dataset == "fss_cell":
        cfg.TRAIN.DATASETS = ('fss_cell',)
        cfg.MODEL.NUM_CLASSES = 14
    elif args.dataset == "coco2017":
        cfg.TRAIN.DATASETS = ('coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TRAIN.DATASETS = ('keypoints_coco_2017_train',)
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError("Unexpected args.dataset: {}".format(args.dataset))
    
    args.cfg_file = "configs/few_shot/e2e_mask_rcnn_R-50-C4_1x_{}.yaml".format(args.group)
    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.SEEN = args.seen

    ### Adaptively adjust some configs ###
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    original_ims_per_batch = cfg.TRAIN.IMS_PER_BATCH
    original_num_gpus = cfg.NUM_GPUS
    if args.batch_size is None:
        args.batch_size = original_batch_size
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS

    print('Adaptive config changes:')
    print('    batch_size: %d --> %d' % (original_batch_size, args.batch_size))
    print('    NUM_GPUS:             %d --> %d' % (original_num_gpus, cfg.NUM_GPUS))
    print('    IMS_PER_BATCH:        %d --> %d' % (original_ims_per_batch, cfg.TRAIN.IMS_PER_BATCH))

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

    ### Adjust learning based on batch size change linearly
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
    print('Adjust BASE_LR linearly according to batch size change: {} --> {}'.format(
        old_base_lr, cfg.SOLVER.BASE_LR))

    # Scale FPN rpn_proposals collect size (post_nms_topN) in `collect` function
    # of `collect_and_distribute_fpn_rpn_proposals.py`
    #
    # post_nms_topN = int(cfg[cfg_key].RPN_POST_NMS_TOP_N * cfg.FPN.RPN_COLLECT_SCALE + 0.5)
    if cfg.FPN.FPN_ON and cfg.MODEL.FASTER_RCNN:
        cfg.FPN.RPN_COLLECT_SCALE = cfg.TRAIN.IMS_PER_BATCH / original_ims_per_batch
        print('Scale FPN rpn_proposals collect size directly propotional to the change of IMS_PER_BATCH:\n'
              '    cfg.FPN.RPN_COLLECT_SCALE: {}'.format(cfg.FPN.RPN_COLLECT_SCALE))
    
    ### Overwrite some solver settings from command line arguments
    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma

    timers = defaultdict(Timer)

    ### Dataset ###
    timers['roidb'].tic()
    imdb, roidb, ratio_list, ratio_index, query = combined_roidb(
        cfg.TRAIN.DATASETS, True)
    timers['roidb'].toc()
    train_size = len(roidb)
    logger.info('{:d} roidb entries'.format(train_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)

    sampler = MinibatchSampler(ratio_list, ratio_index)
    dataset = RoiDataLoader(
        roidb, ratio_list, ratio_index, query, 
        cfg.MODEL.NUM_CLASSES,
        training=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch)

    assert_and_infer_cfg()

    ### Model ###
    maskRCNN = Generalized_RCNN()

    if cfg.CUDA:
        maskRCNN.cuda()

    ### Optimizer ###
    bias_params = []
    nonbias_params = []
    for key, value in dict(maskRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
            else:
                nonbias_params.append(value)
    params = [
        {'params': nonbias_params,
         'lr': cfg.SOLVER.BASE_LR,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_params,
         'lr': cfg.SOLVER.BASE_LR * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0}
    ]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)

    ### Load checkpoint
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])
        if args.resume:
            assert checkpoint['iters_per_epoch'] == train_size // args.batch_size, \
                "iters_per_epoch should match for resume"
            # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
            # However it's fixed on master.
            # optimizer.load_state_dict(checkpoint['optimizer'])
            misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
            if checkpoint['step'] == (checkpoint['iters_per_epoch'] - 1):
                # Resume from end of an epoch
                args.start_epoch = checkpoint['epoch'] + 1
                args.start_iter = 0
            else:
                # Resume from the middle of an epoch.
                # NOTE: dataloader is not synced with previous state
                args.start_epoch = checkpoint['epoch']
                args.start_iter = checkpoint['step'] + 1
        del checkpoint
        torch.cuda.empty_cache()

    if args.load_detectron:  #TODO resume for detectron weights (load sgd momentum values)
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True)

    ### Training Setups ###
    args.run_name = misc_utils.get_run_name()
    output_dir = misc_utils.get_output_dir(args, args.run_name)
    args.cfg_filename = os.path.basename(args.cfg_file)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            #from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            #tblogger = SummaryWriter(output_dir)
            from loggers.logger import Logger
            tblogger = Logger(output_dir)

    ### Training Loop ###
    maskRCNN.train()

    training_stats = TrainingStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.no_save else None)

    iters_per_epoch = int(train_size / args.batch_size)  # drop last
    args.iters_per_epoch = iters_per_epoch
    ckpt_interval_per_epoch = iters_per_epoch // args.ckpt_num_per_epoch
    try:
        logger.info('Training starts !')
        args.step = args.start_iter
        global_step = iters_per_epoch * args.start_epoch + args.step
        for args.epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
            # ---- Start of epoch ----

            # adjust learning rate
            if args.lr_decay_epochs and args.epoch == args.lr_decay_epochs[0] and args.start_iter == 0:
                args.lr_decay_epochs.pop(0)
                net_utils.decay_learning_rate(optimizer, lr, cfg.SOLVER.GAMMA)
                lr *= cfg.SOLVER.GAMMA

            for args.step, input_data in zip(range(args.start_iter, iters_per_epoch), dataloader):

                for key in input_data:
                    if key != 'roidb' and key != 'query': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))
                    if key == 'query':
                        input_data[key] = [list(map(Variable, q)) for q in input_data[key]]

                training_stats.IterTic()
                net_outputs = maskRCNN(**input_data)
                training_stats.UpdateIterStats(net_outputs)
                loss = net_outputs['total_loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_stats.IterToc()

                if (args.step+1) % ckpt_interval_per_epoch == 0:
                    net_utils.save_ckpt(output_dir, args, maskRCNN, optimizer)

                if args.step % args.disp_interval == 0:
                    log_training_stats(training_stats, global_step, lr, input_data)

                global_step += 1

            # ---- End of epoch ----
            # save checkpoint
            net_utils.save_ckpt(output_dir, args, maskRCNN, optimizer)
            # reset starting iter number after first epoch
            args.start_iter = 0

        # ---- Training ends ----
        if iters_per_epoch % args.disp_interval != 0:
            # log last stats at the end
            log_training_stats(training_stats, global_step, lr, input_data)

    except (RuntimeError, KeyboardInterrupt):
        logger.info('Save ckpt on exception ...')
        net_utils.save_ckpt(output_dir, args, maskRCNN, optimizer)
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if args.use_tfboard and not args.no_save:
            tblogger.close()


def log_training_stats(training_stats, global_step, lr, input_data):
    stats = training_stats.GetStats(global_step, lr)
    log_stats(stats, training_stats.misc_args)
    if training_stats.tblogger:
        training_stats.tb_log_stats(stats, global_step)
        training_stats.tblogger._add_images(global_step, input_data)


if __name__ == '__main__':
    main()
