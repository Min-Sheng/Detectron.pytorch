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

"""Representation of the FSS-Cell json dataset format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse

# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

import utils.boxes as box_utils
import utils.segms as segm_utils
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog import ANN_FN
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR
from .dataset_catalog import IM_PREFIX

import pycocotools.mask as mask_util

logger = logging.getLogger(__name__)


class JsonDataset(object):
    """A class representing a FSS-Cell json dataset."""

    def __init__(self, name):
        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        assert os.path.exists(DATASETS[name][ANN_FN]), \
            'Annotation file \'{}\' not found'.format(DATASETS[name][ANN_FN])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.COCO = COCO(DATASETS[name][ANN_FN])
        self.debug_timer = Timer()
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        cats = self.COCO.loadCats(self.COCO.getCatIds())
        categories = [c['name'] for c in cats]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        
        # class name to ind (0~14) 0= __background__
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        # class name to cat_id (1~14)
        self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in cats],
                                                self.COCO.getCatIds())))
        self._init_keypoints()
        
        self.image_index = self._load_image_set_index()

        self.cat_data = {}

        for i in self._class_to_ind.values():
            # i = 1~14
            self.cat_data[i] = []

        # # Set cfg.MODEL.NUM_CLASSES
        # if cfg.MODEL.NUM_CLASSES != -1:
        #     assert cfg.MODEL.NUM_CLASSES == 2 if cfg.MODEL.KEYPOINTS_ON else self.num_classes, \
        #         "number of classes should equal when using multiple datasets"
        # else:
        #     cfg.MODEL.NUM_CLASSES = 2 if cfg.MODEL.KEYPOINTS_ON else self.num_classes

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        return image_ids

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])
    
    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_index[i]

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        file_name = [image_info for image_info in self.COCO.dataset['images'] if image_info['id'] == index][0]['file_name']
        image_path = os.path.join(self.image_directory, file_name)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['boxes', 'segms', 'gt_classes', 'seg_areas', 'gt_overlaps',
                'is_crowd', 'box_to_gt_ind_map', 'gt_cats']
        if self.keypoints is not None:
            keys += ['gt_keypoints', 'has_visible_keypoints']
        return keys

    def get_roidb(self, gt=False):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
        """
        if cfg.DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(self._load_image_set_index()))[:100]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(self._load_image_set_index()))

        for entry in roidb:
            self._prep_roidb_entry(entry)
        
        if gt:
            # Include ground-truth object annotations
            cache_filepath = os.path.join(self.cache_path, self.name+'_gt_roidb.pkl')
            if os.path.exists(cache_filepath) and not cfg.DEBUG:
                self.debug_timer.tic()
                self._add_gt_from_cache(roidb, cache_filepath)
                logger.debug(
                    '_add_gt_from_cache took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
            else:
                self.debug_timer.tic()
                for entry in roidb:
                    self._add_gt_annotations(entry)
                logger.debug(
                    '_add_gt_annotations took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
                if not cfg.DEBUG:
                    with open(cache_filepath, 'wb') as fp:
                        pickle.dump([roidb, self.cat_data], fp, pickle.HIGHEST_PROTOCOL)
                    logger.info('Cache ground truth roidb to %s', cache_filepath)
        _add_class_assignments(roidb)
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(
            self.image_directory, self.image_prefix + entry['file_name']
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['gt_cats'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        #entry['gt_overlaps'] = scipy.sparse.csr_matrix(
        #    np.empty((0, self.num_classes), dtype=np.float32)
        #)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, cfg.MODEL.NUM_CLASSES), dtype=np.float32)
        )
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32
            )
        # Remove unwanted fields that come from the json file (if they exist)
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]

    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'])
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            im_path = self.image_path_from_index(obj['image_id'])
            # crowd regions are RLE encoded and stored as dicts
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
            x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width
            )
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                # Compress the rle
                if 'counts' in obj['segmentation'] and type(obj['segmentation']['counts']) == list:
                    # Magic RLE format handling painfully discovered by looking at the
                    # COCO API showAnns function.
                    rle = mask_util.frPyObjects(obj['segmentation'], height, width)
                    valid_segms.append(rle)
                else:
                    binary_mask = segm_utils.polys_to_mask(obj['segmentation'], height, width)
                    if len(np.unique(binary_mask))==1:
                        continue
                    valid_segms.append(obj['segmentation'])
                valid_objs.append(obj)
                query = {
                        'boxes': obj['clean_bbox'],
                        'segms': valid_segms,
                        'image_path': im_path,
                        'area': obj['area'],
                        }

                self.cat_data[obj['category_id']].append(query)

        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_cats = np.zeros((num_valid_objs), dtype=entry['gt_cats'].dtype)
        #gt_overlaps = np.zeros(
        #    (num_valid_objs, self.num_classes),
        #    dtype=entry['gt_overlaps'].dtype
        #)
        gt_overlaps = np.zeros(
            (num_valid_objs, cfg.MODEL.NUM_CLASSES),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
        )
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype
            )

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = obj['category_id']
            boxes[ix, :] = obj['clean_bbox']
            #gt_classes[ix] = cls
            gt_classes[ix] = 1
            gt_cats[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            #gt_overlaps[ix, cls] = 1.0
            gt_overlaps[ix, 1] = 1.0
        box_utils.validate_boxes(boxes, width=width, height=height)
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['gt_cats'] = np.append(entry['gt_cats'], gt_cats)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0
            )
            entry['has_visible_keypoints'] = im_has_visible_keypoints

    def _add_gt_from_cache(self, roidb, cache_filepath):
        """Add ground truth annotation metadata from cached file."""
        logger.info('Loading cached gt_roidb from %s', cache_filepath)
        with open(cache_filepath, 'rb') as fp:
            [cached_roidb, self.cat_data] = pickle.load(fp)

        assert len(roidb) == len(cached_roidb)

        for entry, cached_entry in zip(roidb, cached_roidb):
            values = [cached_entry[key] for key in self.valid_cached_keys]
            boxes, segms, gt_classes, seg_areas, gt_overlaps, is_crowd, \
                box_to_gt_ind_map, gt_cats = values[:8]
            if self.keypoints is not None:
                gt_keypoints, has_visible_keypoints = values[8:]
            entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            entry['segms'].extend(segms)
            # To match the original implementation:
            # entry['boxes'] = np.append(
            #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
            entry['gt_cats'] = np.append(entry['gt_cats'], gt_cats)
            entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
            entry['box_to_gt_ind_map'] = np.append(
                entry['box_to_gt_ind_map'], box_to_gt_ind_map
            )
            if self.keypoints is not None:
                entry['gt_keypoints'] = np.append(
                    entry['gt_keypoints'], gt_keypoints, axis=0
                )
                entry['has_visible_keypoints'] = has_visible_keypoints

    def _init_keypoints(self):
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
        else:
            return

        # Check if the annotations contain keypoint data or not
        if 'keypoints' in cat_info[0]:
            keypoints = cat_info[0]['keypoints']
            self.keypoints_to_id_map = dict(
                zip(keypoints, range(len(keypoints))))
            self.keypoints = keypoints
            self.num_keypoints = len(keypoints)
            if cfg.KRCNN.NUM_KEYPOINTS != -1:
                assert cfg.KRCNN.NUM_KEYPOINTS == self.num_keypoints, \
                    "number of keypoints should equal when using multiple datasets"
            else:
                cfg.KRCNN.NUM_KEYPOINTS = self.num_keypoints
            self.keypoint_flip_map = {
                'left_eye': 'right_eye',
                'left_ear': 'right_ear',
                'left_shoulder': 'right_shoulder',
                'left_elbow': 'right_elbow',
                'left_wrist': 'right_wrist',
                'left_hip': 'right_hip',
                'left_knee': 'right_knee',
                'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps
    
    def filter(self, roidb):

        # if want to use train_categories, seen = 1 
        # if want to use test_categories , seen = 2
        # if want to use both            , seen = 3

        folds = {
            'all': set(range(1, 15)),
            1: set(range(1, 15)) - set(range(1, 3)),
            2: set(range(1, 15)) - set(range(3, 6)),
            3: set(range(1, 15)) - set(range(6, 9)),
            4: set(range(1, 15)) - set(range(9, 11)),
            5: set(range(1, 15)) - set(range(11, 15)),
        }

        if cfg.SEEN==1:
            self.list = cfg.TRAIN.CATEGORIES
            # Group number to class
            if len(self.list)==1:
                self.list = list(folds[self.list[0]])

        elif cfg.SEEN==2:
            self.list = cfg.TEST.CATEGORIES
            # Group number to class
            if len(self.list)==1:
                self.list = list(folds['all'] - folds[self.list[0]])
        
        elif cfg.SEEN==3:
            self.list = cfg.TRAIN.CATEGORIES + cfg.TEST.CATEGORIES
            # Group number to class
            if len(self.list)==0:
                self.list = list(folds['all'])
        self.inverse_list = self.list
        # Which index need to be remove
        all_index = list(range(len(self.image_index)))

        for index, info in enumerate(roidb):
            for cat in info['gt_cats']:
                if cat in self.list:
                    all_index.remove(index)
                    break

        # Remove index from the end to start
        all_index.reverse()
        for index in all_index:
            self.image_index.pop(index)
            roidb.pop(index)
        return roidb

def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            gt_cats = entry['gt_cats'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['gt_cats'] = np.append(
            entry['gt_cats'],
            np.zeros((num_boxes), dtype=entry['gt_cats'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)
