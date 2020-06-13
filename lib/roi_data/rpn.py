import logging
import numpy as np
import numpy.random as npr

from core.config import cfg
import roi_data.data_utils as data_utils
import utils.blob as blob_utils
import utils.boxes as box_utils
import utils.segms as segm_utils
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def get_rpn_blob_names(is_training=True):
    """Blob names used by RPN."""
    # im_info: (height, width, image scale)
    blob_names = ['im_info']
    if is_training:
        # gt boxes: (batch_idx, x1, y1, x2, y2, cls)
        blob_names += ['roidb']
        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
            # Same format as RPN blobs, but one per FPN level
            for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
                if cfg.TRAIN.RPN_SPATIALLY_REGULARIZED:
                    blob_names += [
                    'rpn_labels_int32_wide_fpn' + str(lvl),
                    'rpn_bbox_targets_wide_fpn' + str(lvl),
                    'rpn_bbox_inside_weights_wide_fpn' + str(lvl),
                    'rpn_bbox_outside_weights_wide_fpn' + str(lvl),
                    'rpn_cls_score_weights_wide_fpn' + str(lvl)
                ]
                else:
                    blob_names += [
                    'rpn_labels_int32_wide_fpn' + str(lvl),
                    'rpn_bbox_targets_wide_fpn' + str(lvl),
                    'rpn_bbox_inside_weights_wide_fpn' + str(lvl),
                    'rpn_bbox_outside_weights_wide_fpn' + str(lvl)
                ]
        else:
            # Single level RPN blobs
            if cfg.TRAIN.RPN_SPATIALLY_REGULARIZED:
                blob_names += [
                    'rpn_labels_int32_wide',
                    'rpn_bbox_targets_wide',
                    'rpn_bbox_inside_weights_wide',
                    'rpn_bbox_outside_weights_wide',
                    'rpn_cls_score_weights_wide'
                ]
            else:
                blob_names += [
                    'rpn_labels_int32_wide',
                    'rpn_bbox_targets_wide',
                    'rpn_bbox_inside_weights_wide',
                    'rpn_bbox_outside_weights_wide'
                ]
    return blob_names


def add_rpn_blobs(blobs, im_scales, roidb):
    """Add blobs needed training RPN-only and end-to-end Faster R-CNN models."""
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
        # RPN applied to many feature levels, as in the FPN paper
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL
        foas = []
        for lvl in range(k_min, k_max + 1):
            field_stride = 2.**lvl
            anchor_sizes = (cfg.FPN.RPN_ANCHOR_START_SIZE * 2.**(lvl - k_min), )
            anchor_aspect_ratios = cfg.FPN.RPN_ASPECT_RATIOS
            foa = data_utils.get_field_of_anchors(
                field_stride, anchor_sizes, anchor_aspect_ratios
            )
            foas.append(foa)
        all_anchors = np.concatenate([f.field_of_anchors for f in foas])
    else:
        foa = data_utils.get_field_of_anchors(cfg.RPN.STRIDE, cfg.RPN.SIZES,
                                              cfg.RPN.ASPECT_RATIOS)
        all_anchors = foa.field_of_anchors

    for im_i, entry in enumerate(roidb):
        segms = [segm[0] for segm in entry['segms']]
        mask = segm_utils.polys_to_mask(segms, entry['height'], entry['width'])
        blobs['binary_mask'] = mask
        scale = im_scales[im_i]
        im_height = np.round(entry['height'] * scale)
        im_width = np.round(entry['width'] * scale)
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0)
        )[0]
        gt_rois = entry['boxes'][gt_inds, :] * scale
        # TODO(rbg): gt_boxes is poorly named;
        # should be something like 'gt_rois_info'
        gt_cats = blob_utils.zeros((len(gt_inds), 1))
        gt_cats = entry['gt_cats'][gt_inds]
        blobs['gt_cats'] = gt_cats
        im_info = np.array([[im_height, im_width, scale]], dtype=np.float32)
        blobs['im_info'].append(im_info)

        # Add RPN targets
        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
            # RPN applied to many feature levels, as in the FPN paper
            rpn_blobs = _get_rpn_blobs(
                im_height, im_width, foas, all_anchors, gt_rois
            )
            for i, lvl in enumerate(range(k_min, k_max + 1)):
                for k, v in rpn_blobs[i].items():
                    blobs[k + '_fpn' + str(lvl)].append(v)
        else:
            # Classical RPN, applied to a single feature level
            rpn_blobs = _get_rpn_blobs(
                im_height, im_width, [foa], all_anchors, gt_rois
            )
            for k, v in rpn_blobs.items():
                blobs[k].append(v)

    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)

    valid_keys = [
        'has_visible_keypoints', 'boxes', 'segms', 'seg_areas', 'gt_classes',
        'gt_overlaps', 'is_crowd', 'box_to_gt_ind_map', 'gt_keypoints', 'gt_cats', 'binary_mask'
    ]
    minimal_roidb = [{} for _ in range(len(roidb))]
    for i, e in enumerate(roidb):
        for k in valid_keys:
            if k in e:
                minimal_roidb[i][k] = e[k]
    # blobs['roidb'] = blob_utils.serialize(minimal_roidb)
    blobs['roidb'] = minimal_roidb

    # Always return valid=True, since RPN minibatches are valid by design
    return True


def _get_rpn_blobs(im_height, im_width, foas, all_anchors, gt_boxes):
    total_anchors = all_anchors.shape[0]
    straddle_thresh = cfg.TRAIN.RPN_STRADDLE_THRESH

    if straddle_thresh >= 0:
        # Only keep anchors inside the image by a margin of straddle_thresh
        # Set TRAIN.RPN_STRADDLE_THRESH to -1 (or a large value) to keep all
        # anchors
        inds_inside = np.where(
            (all_anchors[:, 0] >= -straddle_thresh) &
            (all_anchors[:, 1] >= -straddle_thresh) &
            (all_anchors[:, 2] < im_width + straddle_thresh) &
            (all_anchors[:, 3] < im_height + straddle_thresh)
        )[0]
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
    else:
        inds_inside = np.arange(all_anchors.shape[0])
        anchors = all_anchors
    num_inside = len(inds_inside)

    logger.debug('total_anchors: %d', total_anchors)
    logger.debug('inds_inside: %d', num_inside)
    logger.debug('anchors.shape: %s', str(anchors.shape))

    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_inside, ), dtype=np.int32)
    labels.fill(-1)
    if len(gt_boxes) > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        anchor_by_gt_overlap = box_utils.bbox_overlaps(anchors, gt_boxes)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside),
                                                anchor_to_gt_argmax]

        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax,
            np.arange(anchor_by_gt_overlap.shape[1])
        ]
        # Find all anchors that share the max overlap amount
        # (this includes many ties)
        anchors_with_max_overlap = np.where(
            anchor_by_gt_overlap == gt_to_anchor_max
        )[0]

        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        labels[anchors_with_max_overlap] = 1
        # Fg label: above threshold IOU
        labels[anchor_to_gt_max >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE_PER_IM)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False
        )
        labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]

    # subsample negative labels if we have too many
    # (samples with replacement, but since the set of bg inds is large most
    # samples will not have repeats)
    num_bg = cfg.TRAIN.RPN_BATCH_SIZE_PER_IM - np.sum(labels == 1)
    bg_inds = np.where(anchor_to_gt_max < cfg.TRAIN.RPN_NEGATIVE_OVERLAP)[0]
    if len(bg_inds) > num_bg:
        enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
        labels[enable_inds] = 0
    bg_inds = np.where(labels == 0)[0]

    bbox_targets = np.zeros((num_inside, 4), dtype=np.float32)
    bbox_targets[fg_inds, :] = data_utils.compute_targets(
        anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :]
    )

    # Bbox regression loss has the form:
    #   loss(x) = weight_outside * L(weight_inside * x)
    # Inside weights allow us to set zero loss on an element-wise basis
    # Bbox regression is only trained on positive examples so we set their
    # weights to 1.0 (or otherwise if config is different) and 0 otherwise
    bbox_inside_weights = np.zeros((num_inside, 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = (1.0, 1.0, 1.0, 1.0)

    # The bbox regression loss only averages by the number of images in the
    # mini-batch, whereas we need to average by the total number of example
    # anchors selected
    # Outside weights are used to scale each element-wise loss so the final
    # average over the mini-batch is correct
    bbox_outside_weights = np.zeros((num_inside, 4), dtype=np.float32)
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    bbox_outside_weights[labels == 1, :] = 1.0 / num_examples
    bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

    #===========================================================
    if cfg.TRAIN.RPN_SPATIALLY_REGULARIZED:
        # cls_score_weights: value between [0, 1], 1 is no penalty. 

        cls_score_weights = np.empty((num_inside, ), dtype=np.float32)
        cls_score_weights.fill(1.0)

        im_info = (im_height, im_width)
        cls_score_weights = _calculateGaussianWeights(anchors, gt_boxes, cls_score_weights, labels, im_info)

    #===========================================================

    # Map up to original set of anchors
    labels = data_utils.unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = data_utils.unmap(
        bbox_targets, total_anchors, inds_inside, fill=0
    )
    bbox_inside_weights = data_utils.unmap(
        bbox_inside_weights, total_anchors, inds_inside, fill=0
    )
    bbox_outside_weights = data_utils.unmap(
        bbox_outside_weights, total_anchors, inds_inside, fill=0
    )
    if cfg.TRAIN.RPN_SPATIALLY_REGULARIZED:
        cls_score_weights = data_utils.unmap(cls_score_weights, total_anchors, inds_inside, fill=1.0)

    # Split the generated labels, etc. into labels per each field of anchors
    blobs_out = []
    start_idx = 0
    for foa in foas:
        H = foa.field_size
        W = foa.field_size
        A = foa.num_cell_anchors
        end_idx = start_idx + H * W * A
        _labels = labels[start_idx:end_idx]
        _bbox_targets = bbox_targets[start_idx:end_idx, :]
        _bbox_inside_weights = bbox_inside_weights[start_idx:end_idx, :]
        _bbox_outside_weights = bbox_outside_weights[start_idx:end_idx, :]
        if cfg.TRAIN.RPN_SPATIALLY_REGULARIZED:
            _cls_score_weights = cls_score_weights[start_idx:end_idx]
        start_idx = end_idx

        # labels output with shape (1, A, height, width)
        _labels = _labels.reshape((1, H, W, A)).transpose(0, 3, 1, 2)
        # bbox_targets output with shape (1, 4 * A, height, width)
        _bbox_targets = _bbox_targets.reshape(
            (1, H, W, A * 4)).transpose(0, 3, 1, 2)
        # bbox_inside_weights output with shape (1, 4 * A, height, width)
        _bbox_inside_weights = _bbox_inside_weights.reshape(
            (1, H, W, A * 4)).transpose(0, 3, 1, 2)
        # bbox_outside_weights output with shape (1, 4 * A, height, width)
        _bbox_outside_weights = _bbox_outside_weights.reshape(
            (1, H, W, A * 4)).transpose(0, 3, 1, 2)
        if cfg.TRAIN.RPN_SPATIALLY_REGULARIZED:
            # rpn_cls_score_weight output with shape (1, A, height, width)
            _cls_score_weights = _cls_score_weights.reshape((1, H, W, A)).transpose(0, 3, 1, 2)

        if cfg.TRAIN.RPN_SPATIALLY_REGULARIZED:
            blobs_out.append(
                dict(
                    rpn_labels_int32_wide=_labels,
                    rpn_bbox_targets_wide=_bbox_targets,
                    rpn_bbox_inside_weights_wide=_bbox_inside_weights,
                    rpn_bbox_outside_weights_wide=_bbox_outside_weights,
                    rpn_cls_score_weights_wide=_cls_score_weights
                )
            )
        else:
            blobs_out.append(
                dict(
                    rpn_labels_int32_wide=_labels,
                    rpn_bbox_targets_wide=_bbox_targets,
                    rpn_bbox_inside_weights_wide=_bbox_inside_weights,
                    rpn_bbox_outside_weights_wide=_bbox_outside_weights
                )
            )
    return blobs_out[0] if len(blobs_out) == 1 else blobs_out

def _calculateGaussianWeights(anchors, gt_boxes, cls_score_weights, labels, im_info):
    assert len(labels) == len(cls_score_weights)
    assert anchors.shape[0] == len(cls_score_weights)

    gk_size = cfg.TRAIN.RPN_GAUSSIAN_KERNEL_SIZE # 255 => 0~254
    gk_weights = cfg.TRAIN.RPN_GAUSSIAN_WEIGHTS 
    gkw_normalized_value = cfg.TRAIN.RPN_GAUSSIAN_WEIGHTS_NORMALIZED_VALUE
    for idx, center_anchor in enumerate(anchors):
        if labels[idx] != 1:
            continue
        top_left_x = ((center_anchor[0] + center_anchor[2]) / 2) - ((gk_size - 1) / 2)
        top_left_y = ((center_anchor[1] + center_anchor[3]) / 2) - ((gk_size - 1) / 2)
        bottom_right_x = ((center_anchor[0] + center_anchor[2]) / 2) + ((gk_size - 1) / 2)
        bottom_right_y = ((center_anchor[1] + center_anchor[3]) / 2) + ((gk_size - 1) / 2)

        # get the index where gt_box inside the range
        inside_gk = np.where(
            (gt_boxes[:, 0] >= top_left_x) &
            (gt_boxes[:, 1] >= top_left_y) &
            (gt_boxes[:, 2] <= bottom_right_x) &
            (gt_boxes[:, 3] <= bottom_right_y)
        )[0]
        if inside_gk.size != 0:
            center_anchor_x = (center_anchor[0] + center_anchor[2]) / 2
            center_anchor_y = (center_anchor[1] + center_anchor[3]) / 2
            center_anchor_GKindex_x = (gk_size - 1) / 2 # 127
            center_anchor_GKindex_y = (gk_size - 1) / 2 # 127

            gaussian_value_sum = np.zeros(gk_weights.shape[0])
            for index in inside_gk:
                gt_center_x = (gt_boxes[index, 0] + gt_boxes[index, 2]) / 2
                gt_center_y = (gt_boxes[index, 1] + gt_boxes[index, 3]) / 2
                # Add shift of index by center-anchor's position and ground-truth's position
                gt_center_GKindex_x = center_anchor_GKindex_x + (gt_center_x - center_anchor_x) 
                gt_center_GKindex_y = center_anchor_GKindex_y + (gt_center_y - center_anchor_y)
                # make sure not to out of bounds (0 ~ 254)
                gt_center_GKindex_x = np.min([np.max([int(gt_center_GKindex_x), 0]), gk_size - 1])
                gt_center_GKindex_y = np.min([np.max([int(gt_center_GKindex_y), 0]), gk_size - 1])
                #print((gt_center_GKindex_x, gt_center_GKindex_y))

                # Add wieghts for 4 direction of gaussian kernels
                for gk_idx in range(gk_weights.shape[0]):
                    gaussian_value = gk_weights[gk_idx][gt_center_GKindex_x][gt_center_GKindex_y]
                    gaussian_value_sum[gk_idx] += gaussian_value
            cls_score_weights[idx] = np.min(gaussian_value_sum)

            DEBUG = False
            if DEBUG:
                gaussian_kernel_test(gk_weights, gk_size)
                print(im_info[0], im_info[1])
                img = np.zeros((int(im_info[0]), int(im_info[1])))
                
                cx = int(center_anchor_x)
                cy = int(center_anchor_y)
                s = 2
                for x in range(cx-s, cx+s):
                    for y in range(cy-s ,cy+s):
                        x = np.min([np.max([x, 0]), im_info[1] - 1])
                        y = np.min([np.max([y, 0]), im_info[0] - 1])
                        img[int(y)][int(x)] = 0.5
                
                for i in inside_gk:
                    gx = int((gt_boxes[i, 0] + gt_boxes[i, 2]) / 2)
                    gy = int((gt_boxes[i, 1] + gt_boxes[i, 3]) / 2)
                    
                    s = 2
                    for x in range(gx-s, gx+s):
                        for y in range(gy-s ,gy+s):
                            x = np.min([np.max([x, 0]), im_info[1] - 1])
                            y = np.min([np.max([y, 0]), im_info[0] - 1])
                            img[int(y)][int(x)] = 1
                
                print('Top-left position: ({0}, {1})'.format(top_left_x, top_left_y))
                print('Center of predicted box: ({0}, {1})'.format(center_anchor_x, center_anchor_y))
                print('Bottom-right position: ({0}, {1})'.format(bottom_right_x, bottom_right_y))
                print('Check recptive field size S:')
                print(center_anchor_x - top_left_x, center_anchor_y - top_left_y)
                print(center_anchor_x - bottom_right_x, center_anchor_y - bottom_right_y)
                print('Gaussian weight score of current predicted box: {0}'.format(gaussian_value_sum))
                
                
                for x in range(int(top_left_x), int(bottom_right_x)):
                    x1 = int(np.min([np.max([x, 0]), im_info[1] - 1]))
                    y1 = int(np.min([np.max([top_left_y, 0]), im_info[0] - 1]))
                    y2 = int(np.min([np.max([bottom_right_y, 0]), im_info[0] - 1]))
                    img[y1][x1] = 0.25
                    img[y2][x1] = 0.25
                        
                for y in range(int(top_left_y), int(bottom_right_y)):
                    x1 = int(np.min([np.max([top_left_x, 0]), im_info[1] - 1]))
                    x2 = int(np.min([np.max([bottom_right_x, 0]), im_info[1] - 1]))
                    y1 = int(np.min([np.max([y, 0]), im_info[0] - 1]))
                    img[y1][x1] = 0.25
                    img[y1][x2] = 0.25

                visualGaussianWeight(img, './check.png')

                assert False
    # Normalizing cls_score_weights where the wieght is not 1.
    inds = np.where(cls_score_weights != 1.0)[0]
    if inds.size != 0:
        mean_weight = np.sum(cls_score_weights[inds]) / inds.size
        cls_score_weights[inds] = cls_score_weights[inds] / mean_weight
        #for idx, value in enumerate(cls_score_weights):
        #    cls_score_weights[idx] = 1.0 / np.max([value, np.finfo(float).eps])
    
    # No normalizing
    #for idx, value in enumerate(cls_score_weights):
    #    cls_score_weights[idx] = 1.0 / np.max([value, np.finfo(float).eps])

    return cls_score_weights

def visualGaussianWeight(weightMap, filename):
    weightMap = np.pad(weightMap, ((10, 10), (10, 10)), "constant")
    fig = plt.imshow(weightMap, cmap="jet")
    plt.axis('on') # off
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(True)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    print('Save image: {0}'.format(filename))

def gaussian_kernel_test(gk_weights, size):
    mid = int((size - 1) / 2)
    sum = np.zeros(4)

    # radius 0
    #
    # 0 0 0
    # * * *
    # 0 0 0
    #
    row = mid
    for col in range(size):
        sum[0] += gk_weights[0][row][col]
    
    # radius 45
    #
    # 0 0 *
    # 0 * 0
    # * 0 0
    #
    for row in range(size):
        col = (size - 1) - row
        sum[1] += gk_weights[1][row][col]

    # radius 90
    #
    # 0 * 0
    # 0 * 0
    # 0 * 0
    #
    col = mid
    for row in range(size):
        sum[2] += gk_weights[2][row][col]
    
    # radius 135
    #
    # * 0 0
    # 0 * 0
    # 0 0 *
    #
    for row in range(size):
        col = row
        sum[3] += gk_weights[3][row][col]

    print(sum) # [ 211.42287755  180.4205265   211.42287755  180.4205265 ]