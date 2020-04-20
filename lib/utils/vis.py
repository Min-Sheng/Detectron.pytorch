# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io 
import cv2
from PIL import Image
import numpy as np
import os
import pycocotools.mask as mask_util

from utils.colormap import colormap
import utils.keypoints as keypoint_utils
from core.config import cfg

# Use a non-interactive backend
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pycocotools import mask as COCOmask

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator


_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


def vis_bbox_opencv(img, bbox, thick=1):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')

def save_one_image_gt(im, im_id, output_filename, dataset, dpi=200, draw_bbox=False):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    annIds = dataset.COCO.getAnnIds(imgIds=im_id, catIds=dataset.COCO.getCatIds())
    anns = dataset.COCO.loadAnns(annIds)
    valid_masks = findValidMasks(im, dataset.COCO, anns)
    im = create_grayscale_image(im, np.sum(valid_masks, axis=2)>0)
    ax.imshow(im)
    
    color_list = colormap(rgb=True) / 255

    #dataset.COCO.showAnns(anns)
    #myshowAnns(im, dataset.COCO, anns, draw_bbox=draw_bbox)
    if len(anns) == 0:
        return 0
    if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
        datasetType = 'instances'
    elif 'caption' in anns[0]:
        datasetType = 'captions'
    else:
        raise Exception('datasetType not supported')
    if datasetType == 'instances':
        ax.set_autoscale_on(False)
        polygons = []
        bboxes = []
        color = []
        bboxes_color = []
        mask_color_id = 0
        for ann in anns:
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    t = dataset.COCO.imgs[ann['image_id']]
                    for seg in ann['segmentation']:
                        #poly = np.array(seg).reshape((int(len(seg)/2), 2))
                        #polygons.append(Polygon(poly))
                        #color.append(c)
                        rle = COCOmask.frPyObjects([seg], t['height'], t['width'])
                else:
                    # mask
                    t = dataset.COCO.imgs[ann['image_id']]
                    if type(ann['segmentation']['counts']) == list:
                        rle = COCOmask.frPyObjects([ann['segmentation']], t['height'], t['width'])
                    else:
                        rle = [ann['segmentation']]
                m = COCOmask.decode(rle)
                #img = np.ones( (m.shape[0], m.shape[1], 3) )
                #color_mask = np.random.random((1, 3)).tolist()[0]
                #for i in range(3):
                #    img[:,:,i] = color_mask[i]
                #ax.imshow(np.dstack( (img, m*0.5) ))
                color_mask = color_list[mask_color_id % len(color_list), 0:3]
                mask_color_id += 1
                contour, hier = cv2.findContours(
                    m.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                for c in contour:
                    polygon = Polygon(
                        c.reshape((-1, 2)),
                        fill=True, facecolor=color_mask,
                        edgecolor=color_mask, linewidth=1.0,
                        alpha=0.65)
                    ax.add_patch(polygon)
            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
                sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton'])-1
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]
                for sk in sks:
                    if np.all(v[sk]>0):
                        plt.plot(x[sk],y[sk], linewidth=3, color=c)
                plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)

            if draw_bbox:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                np_poly = np.array(poly).reshape((4,2))
                bboxes.append(Polygon(np_poly))
                #bboxes_color.append(c)
            fig.savefig(output_filename, dpi=dpi)
            plt.close('all')
        #p = PatchCollection(polygons, facecolor=color, linewidths=1.0, alpha=0.7)
        #ax.add_collection(p)
        #colorval = "#%02x%02x%02x" % (110, 255, 0)
        #b = PatchCollection(bboxes, facecolor='none', edgecolors=colorval, linewidths=0.5, alpha=0.6)
        #ax.add_collection(b)
    elif datasetType == 'captions':
        for ann in anns:
            print(ann['caption'])
    fig.savefig(output_filename, dpi=dpi)
    plt.close('all')

def vis_one_image_gt(im, im_id, im_name, output_dir, dataset, dpi=200, ext='png', class_name=None, save=False, draw_bbox=False):
    if save:
        if class_name is not None:
            output_dir = os.path.join(output_dir, class_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    annIds = dataset.COCO.getAnnIds(imgIds=im_id, catIds=dataset.COCO.getCatIds())
    anns = dataset.COCO.loadAnns(annIds)
    valid_masks = findValidMasks(im, dataset.COCO, anns)
    im = create_grayscale_image(im, np.sum(valid_masks, axis=2)>0)
    ax.imshow(im)
    buffer = io.BytesIO()
    #dataset.COCO.showAnns(anns)
    myshowAnns(im, dataset.COCO, anns, draw_bbox=draw_bbox)
    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(buffer, dpi=dpi)
    buffer.seek(0)
    pil_image = Image.open(buffer).convert("RGB")
    if save:
        pil_image.save(os.path.join(output_dir, '{}'.format(output_name)), 'png')
    plt.close('all')
    buffer.close()
    return pil_image

def findValidMasks(im, coco, anns):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    valid_masks = []
    for ann in anns:
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                # polygon
                t = coco.imgs[ann['image_id']]
                for seg in ann['segmentation']:
                    rle = COCOmask.frPyObjects([seg], t['height'], t['width'])
            else:
                # mask
                t = coco.imgs[ann['image_id']]
                if type(ann['segmentation']['counts']) == list:
                    rle = COCOmask.frPyObjects([ann['segmentation']], t['height'], t['width'])
                else:
                    rle = [ann['segmentation']]
            m = COCOmask.decode(rle)
            valid_masks.append(m[:,:,0])
    valid_masks = np.stack(valid_masks, axis=2)
    return valid_masks

def myshowAnns(im, coco, anns, draw_bbox=False):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    color_list = colormap(rgb=True) / 255
    if len(anns) == 0:
        return 0
    if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
        datasetType = 'instances'
    elif 'caption' in anns[0]:
        datasetType = 'captions'
    else:
        raise Exception('datasetType not supported')
    if datasetType == 'instances':
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        bboxes = []
        color = []
        bboxes_color = []
        mask_color_id = 0
        for ann in anns:
            c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    t = coco.imgs[ann['image_id']]
                    for seg in ann['segmentation']:
                        #poly = np.array(seg).reshape((int(len(seg)/2), 2))
                        #polygons.append(Polygon(poly))
                        #color.append(c)
                        rle = COCOmask.frPyObjects([seg], t['height'], t['width'])
                else:
                    # mask
                    t = coco.imgs[ann['image_id']]
                    if type(ann['segmentation']['counts']) == list:
                        rle = COCOmask.frPyObjects([ann['segmentation']], t['height'], t['width'])
                    else:
                        rle = [ann['segmentation']]
                m = COCOmask.decode(rle)
                #img = np.ones( (m.shape[0], m.shape[1], 3) )
                #color_mask = np.random.random((1, 3)).tolist()[0]
                #for i in range(3):
                #    img[:,:,i] = color_mask[i]
                #ax.imshow(np.dstack( (img, m*0.5) ))
                color_mask = color_list[mask_color_id % len(color_list), 0:3]
                mask_color_id += 1
                contour, hier = cv2.findContours(
                    m.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                for c in contour:
                    polygon = Polygon(
                        c.reshape((-1, 2)),
                        fill=True, facecolor=color_mask,
                        edgecolor=color_mask, linewidth=1.0,
                        alpha=0.65)
                    ax.add_patch(polygon)
                        
            if 'keypoints' in ann and type(ann['keypoints']) == list:
                # turn skeleton into zero-based index
                sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton'])-1
                kp = np.array(ann['keypoints'])
                x = kp[0::3]
                y = kp[1::3]
                v = kp[2::3]
                for sk in sks:
                    if np.all(v[sk]>0):
                        plt.plot(x[sk],y[sk], linewidth=3, color=c)
                plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)

            if draw_bbox:
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                np_poly = np.array(poly).reshape((4,2))
                bboxes.append(Polygon(np_poly))
                #bboxes_color.append(c)

        #p = PatchCollection(polygons, facecolor=color, linewidths=1.0, alpha=0.7)
        #ax.add_collection(p)
        #colorval = "#%02x%02x%02x" % (110, 255, 0)
        #b = PatchCollection(bboxes, facecolor='none', edgecolors=colorval, linewidths=0.5, alpha=0.6)
        #ax.add_collection(b)
    elif datasetType == 'captions':
        for ann in anns:
            print(ann['caption'])

def create_grayscale_image(img, mask=None):
    """
    Create a grayscale version of the original image.
    The colors in masked area, if given, will be kept.
    """
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bw = np.stack([img_bw] * 3, axis=2)
    if mask is not None:
        img_bw[mask] = img[mask]
    return img_bw

def save_one_image(
        im, output_filename, boxes, segms=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False, save=False, draw_bbox=False):
    """Visual debugging of detections."""
    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        im = create_grayscale_image(im)
        ax.imshow(im)
        fig.savefig(output_filename, dpi=dpi)
        plt.close('all')
        return

    if segms is not None:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    valid_masks = []
    for i in sorted_inds:
            score = boxes[i, -1]
            if score < thresh:
                continue
            if segms is not None and len(segms) > i:
                e = masks[:, :, i]
                valid_masks.append(e)
    valid_masks = np.stack(valid_masks, axis=2)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    im = create_grayscale_image(im, np.sum(valid_masks, axis=2)>0)
    ax.imshow(im)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        print(dataset.classes[classes[i]], score)
        # show box (off by default, box_alpha=0.0)
        colorval = "#%02x%02x%02x" % (255, 255, 110)
        if draw_bbox:
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                            fill=False, edgecolor=colorval,
                            linewidth=0.8, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            #w_ratio = .4
            #for c in range(3):
            #    color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            #for c in range(3):
            #    img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask, edgecolor=color_mask, linewidth=1.0,
                    alpha=0.65)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = ax.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    ax.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)
                if kps[2, i2] > kp_thresh:
                    ax.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = ax.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = ax.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)

        fig.savefig(output_filename, dpi=dpi)
        plt.close('all')

def vis_one_image(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='png', class_name=None, save=False, draw_bbox=False):
    """Visual debugging of detections."""
    
    if save:
        if class_name is not None:
            output_dir = os.path.join(output_dir, class_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        im = create_grayscale_image(im)
        ax.imshow(im)
        buffer = io.BytesIO()
        output_name = os.path.basename(im_name) + '.' + ext
        fig.savefig(buffer, dpi=dpi)
        buffer.seek(0)
        pil_image = Image.open(buffer).convert("RGB")
        if save:
            pil_image.save(os.path.join(output_dir, '{}'.format(output_name)), 'png')
        plt.close('all')
        buffer.close()
        return pil_image

    if segms is not None:
        masks = mask_util.decode(segms)

    color_list = colormap(rgb=True) / 255

    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    valid_masks = []
    for i in sorted_inds:
            score = boxes[i, -1]
            if score < thresh:
                continue
            if segms is not None and len(segms) > i:
                e = masks[:, :, i]
                valid_masks.append(e)
    valid_masks = np.stack(valid_masks, axis=2)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    im = create_grayscale_image(im, np.sum(valid_masks, axis=2)>0)
    ax.imshow(im)
    buffer = io.BytesIO()

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue
        print(dataset.classes[classes[i]], score)
        # show box (off by default, box_alpha=0.0)
        colorval = "#%02x%02x%02x" % (255, 255, 110)
        if draw_bbox:
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1],
                            fill=False, edgecolor=colorval,
                            linewidth=0.8, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            #w_ratio = .4
            #for c in range(3):
            #    color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            #for c in range(3):
            #    img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask, edgecolor=color_mask, linewidth=1.0,
                    alpha=0.65)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = ax.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    ax.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)
                if kps[2, i2] > kp_thresh:
                    ax.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = ax.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = ax.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)

    output_name = os.path.basename(im_name) + '.' + ext
    fig.savefig(buffer, dpi=dpi)
    buffer.seek(0)
    pil_image = Image.open(buffer).convert("RGB")
    if save:
        pil_image.save(os.path.join(output_dir, '{}'.format(output_name)), 'png')
    plt.close('all')
    buffer.close()
    return pil_image