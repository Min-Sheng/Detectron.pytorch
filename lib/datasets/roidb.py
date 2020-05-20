# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Functions for common roidb manipulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import logging
import numpy as np

import utils.boxes as box_utils
import utils.keypoints as keypoint_utils
import utils.segms as segm_utils
import utils.blob as blob_utils
from core.config import cfg
from .fss_cell import JsonDataset

logger = logging.getLogger(__name__)


def combined_roidb(dataset_names, training=True):
    """Load and concatenate roidbs for one or more datasets, along with optional
    object proposals. The roidb entries are then prepared for use in training,
    which involves caching certain types of metadata for each roidb entry.
    """
    def get_roidb(dataset_name, training):
        imdb = JsonDataset(dataset_name)
        roidb = imdb.get_roidb(gt=True)

        roidb = imdb.filter(roidb)
        if cfg.TRAIN.USE_FLIPPED and training:
            logger.info('Appending horizontally-flipped training examples...')
            extend_with_flipped_entries(roidb, imdb)
        logger.info('Loaded dataset: {:s}'.format(imdb.name))
        return imdb, roidb, imdb.cat_data, imdb.inverse_list

    if isinstance(dataset_names, six.string_types):
        dataset_names = (dataset_names, )
    
    imdbs =[]
    roidbs = []
    querys = []
    reserveds = []
    for dataset_name in dataset_names:
        imdb, roidb, query, reserved = get_roidb(dataset_name, training)
        imdbs.append(imdb)
        roidbs.append(roidb)
        query_filterd = {k: [x for i, x in enumerate(v) if x['area']>300] for k, v in query.items()}
        querys.append(query_filterd)
        reserveds.append(reserved)
    
    imdb = imdbs[0]
    roidb = roidbs[0]
    query = querys[0]
    reserved = reserveds[0]

    if len(roidbs) > 1 and training:
        for r in imdbs[1:]:
            imdb.extend(r)
        for r in roidbs[1:]:
            roidb.extend(r)
        for r in range(len(querys[0])):
            query[r].extend(querys[1][r])
        for r in reserveds[1:]:
            reserved.extend(r)
    if training:
        if cfg.TRAIN.ASPECT_GROUPING or cfg.TRAIN.ASPECT_CROPPING:
        
            roidb = filter_for_training(roidb)
            logger.info('Computing image aspect ratios and ordering the ratios...')
            ratio_list, ratio_index, cat_list = rank_for_training(roidb, reserved)
            logger.info('done')
        else:
            ratio_list, ratio_index = None, None
    else:
        logger.info('Computing image aspect ratios and ordering the ratios...')
        ratio_list, ratio_index, cat_list = test_rank_roidb_ratio(roidb, reserved)
        logger.info('done')

    logger.info('Computing bounding-box regression targets...')
    add_bbox_regression_targets(roidb)
    logger.info('done')

    _compute_and_log_stats(roidb)
    
    return imdb, roidb, ratio_list, ratio_index, query, cat_list


def extend_with_flipped_entries(roidb, dataset):
    """Flip each entry in the given roidb and return a new roidb that is the
    concatenation of the original roidb and the flipped entries.
    "Flipping" an entry means that that image and associated metadata (e.g.,
    ground truth boxes and object proposals) are horizontally flipped.
    """
    flipped_roidb = []
    for entry in roidb:
        width = entry['width']
        boxes = entry['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        flipped_entry = {}
        dont_copy = ('boxes', 'segms', 'gt_keypoints', 'flipped')
        for k, v in entry.items():
            if k not in dont_copy:
                flipped_entry[k] = v
        flipped_entry['boxes'] = boxes
        flipped_entry['segms'] = segm_utils.flip_segms(
            entry['segms'], entry['height'], entry['width']
        )
        if dataset.keypoints is not None:
            flipped_entry['gt_keypoints'] = keypoint_utils.flip_keypoints(
                dataset.keypoints, dataset.keypoint_flip_map,
                entry['gt_keypoints'], entry['width']
            )
        flipped_entry['flipped'] = True
        flipped_roidb.append(flipped_entry)
    roidb.extend(flipped_roidb)


def filter_for_training(roidb):
    """Remove roidb entries that have no usable RoIs based on config settings.
    """
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        if cfg.MODEL.KEYPOINTS_ON:
            # If we're training for keypoints, exclude images with no keypoints
            valid = valid and entry['has_visible_keypoints']
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    logger.info('Filtered {} roidb entries: {} -> {}'.
                format(num - num_after, num, num_after))
    return filtered_roidb


def rank_for_training(roidb, reserved):
    """Rank the roidb entries according to image aspect ration and mark for cropping
    for efficient batching if image is too long.
    Returns:
        ratio_list: ndarray, list of aspect ratios from small to large
        ratio_index: ndarray, list of roidb entry indices correspond to the ratios
    """
    RATIO_HI = cfg.TRAIN.ASPECT_HI  # largest ratio to preserve.
    RATIO_LO = cfg.TRAIN.ASPECT_LO  # smallest ratio to preserve.

    need_crop_cnt = 0

    ratio_list = []
    cat_list = [] # category list reserved

    for i, entry in enumerate(roidb):
        width = entry['width']
        height = entry['height']
        ratio = width / float(height)

        if cfg.TRAIN.ASPECT_CROPPING:
            if ratio > RATIO_HI:
                entry['need_crop'] = True
                ratio = RATIO_HI
                need_crop_cnt += 1
            elif ratio < RATIO_LO:
                entry['need_crop'] = True
                ratio = RATIO_LO
                need_crop_cnt += 1
            else:
                entry['need_crop'] = False
        else:
            entry['need_crop'] = False

        ratio_list.append(ratio)
        for j in np.unique(entry['gt_cats']):
            if j in reserved:
                cat_list.append(j)

    if cfg.TRAIN.ASPECT_CROPPING:
        logging.info('Number of entries that need to be cropped: %d. Ratio bound: [%.2f, %.2f]',
                     need_crop_cnt, RATIO_LO, RATIO_HI)
    ratio_list = np.array(ratio_list)
    cat_list = np.array(cat_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index, cat_list[ratio_index]

def test_rank_roidb_ratio(roidb, reserved):
    # rank roidb based on the ratio between width and height.
    RATIO_HI = cfg.TEST.ASPECT_HI  # largest ratio to preserve.
    RATIO_LO = cfg.TEST.ASPECT_LO  # smallest ratio to preserve.
    
    need_crop_cnt = 0

    # Image can show more than one time for test different category 
    ratio_list = []
    ratio_index = [] # image index reserved
    cat_list = [] # category list reserved

    for i, entry in enumerate(roidb):
        width = entry['width']
        height = entry['height']
        ratio = width / float(height)

        if cfg.TEST.ASPECT_CROPPING:
            if ratio > RATIO_HI:
                entry['need_crop'] = True
                ratio = RATIO_HI
                need_crop_cnt += 1
            elif ratio < RATIO_LO:
                entry['need_crop'] = True
                ratio = RATIO_LO
                need_crop_cnt += 1
            else:
                entry['need_crop'] = False
        else:
            entry['need_crop'] = False

        for j in np.unique(entry['gt_cats']):
            if j in reserved:
                ratio_list.append(ratio)
                ratio_index.append(i)
                cat_list.append(j)

    if cfg.TEST.ASPECT_CROPPING:
        logging.info('Number of entries that need to be cropped: %d. Ratio bound: [%.2f, %.2f]',
                     need_crop_cnt, RATIO_LO, RATIO_HI)

    ratio_list = np.array(ratio_list)
    ratio_index = np.array(ratio_index)
    cat_list = np.array(cat_list)

    return ratio_list, ratio_index, cat_list

def test_cat_list(roidb, reserved):
    cat_list = [] # category list reserved
    for i, entry in enumerate(roidb):
        for j in np.unique(entry['gt_cats']):
            if j in reserved:
                cat_list.append(j)
    cat_list = np.array(cat_list)
    return cat_list

def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    for entry in roidb:
        entry['bbox_targets'] = _compute_targets(entry)


def _compute_targets(entry):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = entry['boxes']
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th)
    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils.bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]
    # Use class "1" for all boxes if using class_agnostic_bbox_reg
    targets[ex_inds, 0] = (
        1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else labels[ex_inds])
    targets[ex_inds, 1:] = box_utils.bbox_transform_inv(
        ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    return targets


def _compute_and_log_stats(roidb):
    classes = roidb[0]['dataset'].classes
    char_len = np.max([len(c) for c in classes])
    hist_bins = np.arange(len(classes) + 1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((len(classes)), dtype=np.int)
    for entry in roidb:
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_classes = entry['gt_classes'][gt_inds]
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    logger.debug('Ground-truth class histogram:')
    for i, v in enumerate(gt_hist):
        logger.debug(
            '{:d}{:s}: {:d}'.format(
                i, classes[i].rjust(char_len), v))
    logger.debug('-' * char_len)
    logger.debug(
        '{:s}: {:d}'.format(
            'total'.rjust(char_len), np.sum(gt_hist)))