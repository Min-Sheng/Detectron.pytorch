import random
import math
import numpy as np
import numpy.random as npr
from collections import Counter
import cv2
from scipy.misc import imread


import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import default_collate
from torch._six import int_classes as _int_classes

from core.config import cfg
from roi_data.minibatch import get_minibatch
import utils.blob as blob_utils
# from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes


class RoiDataLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, query, num_classes, training=True, cat_list=None, shot=1):
        self.shot =shot
        self._roidb = roidb
        self._query = query
        self._num_classes = num_classes
        self.training = training
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.cat_list = cat_list
        self.data_size = len(self.ratio_index)
        self.query_position = 0

        self.filter()
        self.probability()

    def __getitem__(self, index_tuple):
        index, ratio = index_tuple
        single_db = [self._roidb[index]]
        blobs, valid = get_minibatch(single_db)
        #TODO: Check if minibatch is valid ? If not, abandon it.
        # Need to change _worker_loop in torch.utils.data.dataloader.py.

        # Squeeze batch dim
        for key in blobs:
            if key != 'roidb'  and key != 'gt_cats' and key != 'binary_mask':
                blobs[key] = blobs[key].squeeze(axis=0)

        blobs['gt_cats'] = [x for x in blobs['gt_cats'] if x in self.list]
        blobs['gt_cats'] = np.array(blobs['gt_cats'])

        scale = blobs['im_info'][-1]
        mask = cv2.resize(blobs['binary_mask'], None, None, fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)
        blobs['binary_mask'] = mask
        query_type = 1
        
        if self.training:
            # Random choice query catgory
            positive_catgory = blobs['gt_cats']
            negative_catgory = np.array(list(set(self.cat_list) - set(positive_catgory)))

            r = random.random()
            if r < cfg.TRAIN.QUERY_POSITIVE_RATE:
                query_type = 1
                cand = np.unique(positive_catgory)
                if len(cand)==1:
                    choice = cand[0]
                else:
                    p = []
                    for i in cand:
                        p.append(self.show_time[i])
                    p = np.array(p)
                    p /= p.sum()
                    choice  = np.random.choice(cand,1,p=p)[0]
                query = self.load_query(choice)
            elif r >= cfg.TRAIN.QUERY_POSITIVE_RATE and r < cfg.TRAIN.QUERY_POSITIVE_RATE + cfg.TRAIN.QUERY_GLOBAL_NEGATIVE_RATE:
                query_type = 0
                im = blobs['data'].copy()
                binary_mask = blobs['binary_mask'].copy()
                patch = self.sample_bg(im, binary_mask)
                if len(patch) == self.shot:
                    query = patch
                else:
                    query_type = 0
                    cand = negative_catgory
                    choice  = np.random.choice(cand,1)[0]
                    query = self.load_query(choice)
            else:
                query_type = 0
                cand = negative_catgory
                choice  = np.random.choice(cand,1)[0]
                query = self.load_query(choice)

        else:
            query = self.load_query(index, single_db[0]['id'])

        blobs['query'] = query
        blobs['query_type'] = query_type

        if 'gt_cats' in blobs: 
            del blobs['gt_cats']
        if 'binary_mask' in blobs: 
            del blobs['binary_mask']
            
        if self.training:
            if self._roidb[index]['need_crop']:
                self.crop_data(blobs, ratio)
                # Check bounding box
                entry = blobs['roidb'][0]
                boxes = entry['boxes']
                invalid = (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])
                valid_inds = np.nonzero(~ invalid)[0]
                if len(valid_inds) < len(boxes):
                    for key in ['boxes', 'gt_classes', 'seg_areas', 'gt_overlaps', 'is_crowd',
                                'box_to_gt_ind_map', 'gt_keypoints']:
                        if key in entry:
                            entry[key] = entry[key][valid_inds]
                    entry['segms'] = [entry['segms'][ind] for ind in valid_inds]

            blobs['roidb'] = blob_utils.serialize(blobs['roidb'])  # CHECK: maybe we can serialize in collate_fn

            return blobs
        else:
            blobs['roidb'] = blob_utils.serialize(blobs['roidb'])
            choice = self.cat_list[index]
            blobs['choice'] = choice
            return blobs

    def sample_bg(self, im, mask, patch_size=128, T=10000):
        _, height, width = im.shape
        t = 0
        n = 0
        patches = []
        while n < self.shot and t < T:
            random_height = np.random.randint(0, height - patch_size - 1)
            random_width = np.random.randint(0, width - patch_size - 1)
            im_patch = im[:, random_height:random_height+patch_size, random_width:random_width+patch_size]
            mask_patch = mask[random_height:random_height+patch_size, random_width:random_width+patch_size].astype(np.bool)
            if np.count_nonzero(~mask_patch):
                im_patch = im_patch.transpose((1, 2, 0))
                if random.randint(0,99)/100 > 0.5:
                    im_patch = im_patch[:, ::-1, :]
                if patch_size > 64:
                    im_patch = cv2.resize(im_patch, (64, 64), interpolation=cv2.INTER_LINEAR)
                patches.append(im_patch.transpose((2, 0, 1)).copy())
                n = n + 1
            t = t + 1
        return patches

    def crop_data(self, blobs, ratio):
        data_height, data_width = map(int, blobs['im_info'][:2])
        boxes = blobs['roidb'][0]['boxes']
        if ratio < 1:  # width << height, crop height
            size_crop = math.ceil(data_width / ratio)  # size after crop
            min_y = math.floor(np.min(boxes[:, 1]))
            max_y = math.floor(np.max(boxes[:, 3]))
            box_region = max_y - min_y + 1
            if min_y == 0:
                y_s = 0
            else:
                if (box_region - size_crop) < 0:
                    y_s_min = max(max_y - size_crop, 0)
                    y_s_max = min(min_y, data_height - size_crop)
                    y_s = y_s_min if y_s_min == y_s_max else \
                        npr.choice(range(y_s_min, y_s_max + 1))
                else:
                    # CHECK: rethinking the mechnism for the case box_region > size_crop
                    # Now, the crop is biased on the lower part of box_region caused by
                    # // 2 for y_s_add
                    y_s_add = (box_region - size_crop) // 2
                    y_s = min_y if y_s_add == 0 else \
                        npr.choice(range(min_y, min_y + y_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, y_s:(y_s + size_crop), :,]
            # Update im_info
            blobs['im_info'][0] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 1] -= y_s
            boxes[:, 3] -= y_s
            np.clip(boxes[:, 1], 0, size_crop - 1, out=boxes[:, 1])
            np.clip(boxes[:, 3], 0, size_crop - 1, out=boxes[:, 3])
            blobs['roidb'][0]['boxes'] = boxes
        else:  # width >> height, crop width
            size_crop = math.ceil(data_height * ratio)
            min_x = math.floor(np.min(boxes[:, 0]))
            max_x = math.floor(np.max(boxes[:, 2]))
            box_region = max_x - min_x + 1
            if min_x == 0:
                x_s = 0
            else:
                if (box_region - size_crop) < 0:
                    x_s_min = max(max_x - size_crop, 0)
                    x_s_max = min(min_x, data_width - size_crop)
                    x_s = x_s_min if x_s_min == x_s_max else \
                        npr.choice(range(x_s_min, x_s_max + 1))
                else:
                    x_s_add = (box_region - size_crop) // 2
                    x_s = min_x if x_s_add == 0 else \
                        npr.choice(range(min_x, min_x + x_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, :, x_s:(x_s + size_crop)]
            # Update im_info
            blobs['im_info'][1] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 0] -= x_s
            boxes[:, 2] -= x_s
            np.clip(boxes[:, 0], 0, size_crop - 1, out=boxes[:, 0])
            np.clip(boxes[:, 2], 0, size_crop - 1, out=boxes[:, 2])
            blobs['roidb'][0]['boxes'] = boxes

    def load_query(self, choice, id=0, aug=True):
        
        if self.training:
            # Random choice query catgory image
            all_data = self._query[choice]
            k_shot_data = random.sample(all_data, self.shot)
        else:
            # Take out the purpose category for testing
            catgory = self.cat_list[choice]
            # list all the candidate image 
            all_data = self._query[catgory]

            # Use image_id to determine the random seed
            # The list l is candidate sequence, which random by image_id
            random.seed(id)
            l = list(range(len(all_data)))
            random.shuffle(l)

            # choose the candidate sequence and take out the data information
            position=[l[(self.query_position+i)%len(l)] for i in range(self.shot)]
            k_shot_data = [all_data[i] for i in position]
        
        query = []
        for data in k_shot_data:
            # Get image
            path = data['image_path']
            #im = cv2.imread(path)
            im = imread(path)
        

            if len(im.shape) == 2:
                im = im[:,:,np.newaxis]
                im = np.concatenate((im,im,im), axis=2)

            if self.training and aug:
                def box_aug(q_h, q_w, x1, y1, x2, y2, p=0.35):
                    h, w = y2 - y1, x2 - x1
                    cty, ctx = h / 2 + y1, w / 2 + x1
                    new_h = (1 + np.random.rand() * p) * h
                    new_w = (1 + np.random.rand() * p) * w
                    #new_h = (1 + p) * h
                    #new_w = (1 + p) * w
                    new_x1, new_x2 = max(0, ctx - new_w / 2), min(q_w - 1, ctx + new_w / 2)
                    new_y1, new_y2 = max(0, cty - new_h / 2), min(q_h - 1, cty + new_h / 2)
                    return  new_x1, new_y1, new_x2, new_y2
                q_h, q_w, q_c = im.shape
                x1, y1, x2, y2 = data['boxes']
                data['boxes'] = box_aug(q_h, q_w, x1, y1, x2, y2, p=cfg.TRAIN.QUERY_BOX_AUG)
                
            im = blob_utils.crop(im, data['boxes'], cfg.TRAIN.QUERY_SIZE)
            # flip the channel, since the original one using cv2
            # rgb -> bgr
            # im = im[:,:,::-1]
            if random.randint(0,99)/100 > 0.5 and self.training:
                im = im[:, ::-1, :]
                        
            im, im_scale = blob_utils.prep_im_for_blob(im, cfg.PIXEL_MEANS, [cfg.TRAIN.QUERY_SIZE],
                        cfg.TRAIN.MAX_SIZE)
            query.append(blob_utils.im_list_to_blob(im).squeeze(0))

        return query
    
    def __len__(self):
        return self.data_size
    
    def filter(self):

        folds = {
            'all': set(range(1, 15)),
            1: set(range(1, 15)) - set(range(1, 3)),
            2: set(range(1, 15)) - set(range(3, 6)),
            3: set(range(1, 15)) - set(range(6, 9)),
            4: set(range(1, 15)) - set(range(9, 11)),
            5: set(range(1, 15)) - set(range(11, 15))
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

    def probability(self):
        show_time = {}
        for i in self.list:
            show_time[i] = 0
        for roi in self._roidb:
            result = Counter(roi['gt_cats'])
            for t in result:
                if t in self.list:
                    show_time[t] += result[t]

        for i in self.list:
            show_time[i] = 1/show_time[i]

        sum_prob = sum(show_time.values())

        for i in self.list:
            show_time[i] = show_time[i]/sum_prob
        
        self.show_time = show_time



def cal_minibatch_ratio(ratio_list):
    """Given the ratio_list, we want to make the RATIO same for each minibatch on each GPU.
    Note: this only work for 1) cfg.TRAIN.MAX_SIZE is ignored during `prep_im_for_blob` 
    and 2) cfg.TRAIN.SCALES containing SINGLE scale.
    Since all prepared images will have same min side length of cfg.TRAIN.SCALES[0], we can
     pad and batch images base on that.
    """
    DATA_SIZE = len(ratio_list)
    ratio_list_minibatch = np.empty((DATA_SIZE,))
    num_minibatch = int(np.ceil(DATA_SIZE / cfg.TRAIN.IMS_PER_BATCH))  # Include leftovers
    for i in range(num_minibatch):
        left_idx = i * cfg.TRAIN.IMS_PER_BATCH
        right_idx = min((i+1) * cfg.TRAIN.IMS_PER_BATCH - 1, DATA_SIZE - 1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        ratio_list_minibatch[left_idx:(right_idx+1)] = target_ratio
    return ratio_list_minibatch


class MinibatchSampler(torch_sampler.Sampler):
    def __init__(self, ratio_list, ratio_index, shuffle=True):
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.num_data = len(ratio_list)
        self.shuffle = shuffle

        if cfg.TRAIN.ASPECT_GROUPING:
            # Given the ratio_list, we want to make the ratio same
            # for each minibatch on each GPU.
            self.ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

    def __iter__(self):
        if cfg.TRAIN.ASPECT_GROUPING:
            # indices for aspect grouping awared permutation
            n, rem = divmod(self.num_data, cfg.TRAIN.IMS_PER_BATCH)
            round_num_data = n * cfg.TRAIN.IMS_PER_BATCH
            indices = np.arange(round_num_data)
            if self.shuffle:
                npr.shuffle(indices.reshape(-1, cfg.TRAIN.IMS_PER_BATCH))  # inplace shuffle
            if rem != 0:
                indices = np.append(indices, np.arange(round_num_data, round_num_data + rem))
            ratio_index = self.ratio_index[indices]
            ratio_list_minibatch = self.ratio_list_minibatch[indices]
        else:
            rand_perm = npr.permutation(self.num_data)
            ratio_list = self.ratio_list[rand_perm]
            ratio_index = self.ratio_index[rand_perm]
            # re-calculate minibatch ratio list
            ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

        return iter(zip(ratio_index.tolist(), ratio_list_minibatch.tolist()))

    def __len__(self):
        return self.num_data


class BatchSampler(torch_sampler.BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, torch_sampler.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)  # Difference: batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size



def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    list_of_roidb = [blobs.pop('roidb') for blobs in list_of_blobs]
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = list_of_blobs[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        # Pad image data
        mini_list = pad_image_data(mini_list)
        minibatch = default_collate(mini_list)
        minibatch['roidb'] = list_of_roidb[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        for key in minibatch:
            Batch[key].append(minibatch[key])

    return Batch


def pad_image_data(list_of_blobs):
    max_shape = blob_utils.get_max_shape([blobs['data'].shape[1:] for blobs in list_of_blobs])
    output_list = []
    for blobs in list_of_blobs:
        data_padded = np.zeros((3, max_shape[0], max_shape[1]), dtype=np.float32)
        _, h, w = blobs['data'].shape
        data_padded[:, :h, :w] = blobs['data']
        blobs['data'] = data_padded
        output_list.append(blobs)
    return output_list