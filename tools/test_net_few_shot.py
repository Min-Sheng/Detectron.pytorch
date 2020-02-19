"""Perform inference on one or more datasets."""

import argparse
import cv2
from scipy.misc import imread
from PIL import Image
import os
import yaml
import datetime
import pprint
import sys
import time
from collections import defaultdict

import torch
import numpy as np

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from datasets import task_evaluation
from core.test_few_shot import im_detect_all
from core.test_engine import extend_results
from datasets.roidb import combined_roidb
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
import utils.logging
import utils.vis as vis_utils
from utils.io import save_object

from utils.timer import Timer

from core.test_engine import initialize_model_from_cfg, empty_results

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument('--g', dest='group',
                      help='which group to train',
                      default=1)
    parser.add_argument('--seen', dest='seen',default=1, type=int)
    #parser.add_argument(
    #    '--cfg', dest='cfg_file', required=True,
    #    help='optional config file')
    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')
    parser.add_argument(
        '--a', dest='average', help='average the top_k candidate samples', default=1, type=int)
    return parser.parse_args()

def main():
    """Main function"""
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    args.cfg_file = "configs/few_shot/e2e_mask_rcnn_R-50-C4_1x_{}.yaml".format(args.group)

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    cfg.VIS = args.vis
    cfg.SEEN = int(args.seen)

    if args.dataset == "fss_cell":
        cfg.TEST.DATASETS = ('fss_cell',)
        cfg.MODEL.NUM_CLASSES = 14
    elif args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 2
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    #logger.info('Testing with config:')
    #logger.info(pprint.pformat(cfg))

    # manually set args.cuda
    args.cuda = True

    timer_for_ds = defaultdict(Timer)

    ### Dataset ###
    timer_for_ds['roidb'].tic()
    imdb, roidb, ratio_list, ratio_index, query = combined_roidb(
        cfg.TEST.DATASETS, [], False)
    timer_for_ds['roidb'].toc()
    roidb_size = len(roidb)
    logger.info('{:d} roidb entries'.format(roidb_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timer_for_ds['roidb'].average_time)

    batchSampler = BatchSampler(
        sampler=MinibatchSampler(ratio_list, ratio_index[0], shuffle=False),
        batch_size=1,
        drop_last=False,
    )
    dataset = RoiDataLoader(
        roidb, ratio_list, ratio_index, query, 
        cfg.MODEL.NUM_CLASSES,
        training=False)
    
    ### Model ###
    model = initialize_model_from_cfg(args, gpu_id=0)

    timer_for_total = defaultdict(Timer)
    timer_for_total['total_test_time'].tic()
    for avg in range(args.average):
        dataset.query_position = avg
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_sampler=batchSampler, 
            num_workers=cfg.DATA_LOADER.NUM_THREADS,
            collate_fn=collate_minibatch
        )
        dataiterator = iter(dataloader)

        num_images = len(ratio_index[0])
        num_classes = cfg.MODEL.NUM_CLASSES
        all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
        
        # total quantity of testing images
        num_detect = len(ratio_index[0])

        timers = defaultdict(Timer)
        for i, index in enumerate(ratio_index[0]):
            input_data = next(dataiterator)
            entry = dataset._roidb[dataset.ratio_index[i]]
            if cfg.TEST.PRECOMPUTED_PROPOSALS:
                # The roidb may contain ground-truth rois (for example, if the roidb
                # comes from the training or val split). We only want to evaluate
                # detection on the *non*-ground-truth rois. We select only the rois
                # that have the gt_classes field set to 0, which means there's no
                # ground truth.
                box_proposals = entry['boxes'][entry['gt_classes'] == 0]
                if len(box_proposals) == 0:
                    continue
            else:
                # Faster R-CNN type models generate proposals on-the-fly with an
                # in-network RPN; 1-stage models don't require proposals.
                box_proposals = None
            
            #im = cv2.imread(entry['image'])
            im = imread(entry['image'])
            
            if len(im.shape) == 2:
                im = im[:,:,np.newaxis]
                im = np.concatenate((im,im,im), axis=2)
            
            cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(model, im, input_data['query'], box_proposals, timers)

            extend_results(i, all_boxes, cls_boxes_i)
            if cls_segms_i is not None:
                extend_results(i, all_segms, cls_segms_i)
            if cls_keyps_i is not None:
                extend_results(i, all_keyps, cls_keyps_i)
            
            if i % 10 == 0:  # Reduce log file size
                ave_total_time = np.sum([t.average_time for t in timers.values()])
                eta_seconds = ave_total_time * (num_images - i - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                det_time = (
                    timers['im_detect_bbox'].average_time +
                    timers['im_detect_mask'].average_time +
                    timers['im_detect_keypoints'].average_time
                )
                misc_time = (
                    timers['misc_bbox'].average_time +
                    timers['misc_mask'].average_time +
                    timers['misc_keypoints'].average_time
                )
                logger.info(
                    (
                        'im_detect: range [{:d}, {:d}] of {:d}: '
                        '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                    ).format(
                        1, num_detect, num_detect, i + 1,
                        num_detect, det_time, misc_time, eta
                    )
                )

            if cfg.VIS:
                im_name = entry['image']
                class_name = im_name.split('/')[-4]
                file_name = im_name.split('/')[-3]

                vis_utils.vis_one_image(
                    im,
                    '{:d}_{:s}'.format(i, file_name),
                    os.path.join(args.output_dir, 'vis'),
                    cls_boxes_i,
                    segms = cls_segms_i,
                    keypoints = cls_keyps_i,
                    thresh = cfg.VIS_TH,
                    box_alpha = 0.8,
                    dataset = imdb,
                    show_class = False,
                    class_name = class_name,
                    query = input_data['query']
                )
            
            """
            if cfg.VIS:
                
                im_name = dataset._roidb[dataset.ratio_index[i]]['image']
                class_name = im_name.split('/')[-4]
                file_name = im_name.split('/')[-3]
                im2show = Image.open(im_name).convert("RGB")
                o_query = input_data['query'][0][0][0].permute(1, 2, 0).contiguous().cpu().numpy()
                o_query *= [0.229, 0.224, 0.225]
                o_query += [0.485, 0.456, 0.406]
                o_query *= 255
                o_query = o_query[:,:,::-1]

                o_query = Image.fromarray(o_query.astype(np.uint8))
                query_w, query_h = o_query.size
                query_bg = Image.new('RGB', (im2show.size), (255, 255, 255))
                bg_w, bg_h = query_bg.size
                offset = ((bg_w - query_w) // 2, (bg_h - query_h) // 2)
                query_bg.paste(o_query, offset)
                
                im2show = np.asarray(im2show)
                o_query = np.asarray(query_bg)

                im2show = np.concatenate((im2show, o_query), axis=1)
                output_dir = os.path.join(args.output_dir, 'vis')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                im_save_dir = os.path.join(output_dir, class_name)
                if not os.path.exists(im_save_dir):
                    os.makedirs(im_save_dir)
                im_save_name = os.path.join(im_save_dir, file_name + '_%d_d.png'%(i))
                cv2.imwrite(im_save_name, cv2.cvtColor(im2show, cv2.COLOR_RGB2BGR))
                """
        cfg_yaml = yaml.dump(cfg)
        det_name = 'detections.pkl'
        det_file = os.path.join(args.output_dir, det_name)
        save_object(
            dict(
                all_boxes=all_boxes,
                all_segms=all_segms,
                all_keyps=all_keyps,
                cfg=cfg_yaml
            ), det_file
        )
        logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
        
        all_results = task_evaluation.evaluate_all(
            imdb, all_boxes, all_segms, all_keyps, args.output_dir
        )
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)
    
    timer_for_total['total_test_time'].toc()
    logger.info('Total inference time: {:.3f}s'.format(timer_for_total['total_test_time'].average_time))

if __name__ == '__main__':
    main()