# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import cv2

DEBUG = False

class ProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # labels
        top[1].reshape(1, 1)
        # bbox_targets
        top[2].reshape(1, self._num_classes * 4)
        # bbox_inside_weights
        top[3].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[4].reshape(1, self._num_classes * 4)

    def forward(self, bottom, top):
        #print 'forward proposal target'
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        num_images = len(np.unique(all_rois[:, 0]))
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        all_gt_boxes = bottom[1].data
        all_flags = bottom[2].data

        # Sanity check: single batch only
        # assert np.all(all_rois[:, 0] == 0), \
        #         'Only single item batches are supported'
        # num_images = 1
        rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        top_all_rois = np.empty((0, 5), dtype=all_rois.dtype)
        top_all_labels = np.empty((0), dtype=all_gt_boxes.dtype)
        top_all_bbox_targets = np.empty((0, self._num_classes * 4), dtype=np.float32)
        top_all_bbox_inside_weights = np.empty((0, self._num_classes * 4), dtype=np.float32)
        for i in xrange(num_images):
            batch_gt_boxes_inds = np.where(all_gt_boxes[:, 5] == i)[0]
            batch_gt_boxes = all_gt_boxes[batch_gt_boxes_inds, 0:-1]
            batch_flags_inds = np.where(all_flags[:, 1] == i)[0]
            batch_flags = all_flags[batch_flags_inds, 0]
            # Include ground-truth boxes in the set of candidate rois
            single_all_rois_inds = np.where(all_rois[:, 0] == i)[0]
            single_all_rois = all_rois[single_all_rois_inds]
            ones = np.ones((batch_gt_boxes.shape[0], 1), dtype=batch_gt_boxes.dtype)
            single_all_rois = np.vstack(
                (single_all_rois, np.hstack((ones * i, batch_gt_boxes[:, :-1])))
            )

            # Sample rois with classification labels and bounding box regression
            # targets
            labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
                single_all_rois, batch_gt_boxes, batch_flags, fg_rois_per_image,
                rois_per_image, self._num_classes, i)

            # for j in xrange(len(batch_gt_boxes)):
            #     [x1, y1, x2, y2] = batch_gt_boxes[j, :-1].astype(np.int32)
            #     cv2.rectangle(cfg.TRAIN.IMAGES_LIST[i], (x1, y1), (x2, y2), (0, 255, 255), 5)
            # false_rois = rois[np.where(labels == 0)[0]]
            # for j in xrange(len(false_rois)):
            #     [x1, y1, x2, y2] = false_rois[j, 1:].astype(np.int32)
            #     cv2.rectangle(cfg.TRAIN.IMAGES_LIST[i], (x1, y1), (x2, y2), (255, 255, 0), 1)
            # for label_i in xrange(self._num_classes):
            #     true_rois = rois[np.where(labels == label_i + 1)[0]]
            #     for j in xrange(len(true_rois)):
            #         [x1, y1, x2, y2] = true_rois[j, 1:].astype(np.int32)
            #         cv2.rectangle(cfg.TRAIN.IMAGES_LIST[i], (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.imshow('test_im', cfg.TRAIN.IMAGES_LIST[i])
            # cv2.waitKey()

            top_all_rois = np.vstack((top_all_rois, rois))
            top_all_labels = np.concatenate((top_all_labels, labels))
            top_all_bbox_targets = np.vstack((top_all_bbox_targets, bbox_targets))
            top_all_bbox_inside_weights = np.vstack((top_all_bbox_inside_weights, bbox_inside_weights))

            if DEBUG:
                print 'num fg: {}'.format((labels > 0).sum())
                print 'num bg: {}'.format((labels == 0).sum())
                self._count += 1
                self._fg_num += (labels > 0).sum()
                self._bg_num += (labels == 0).sum()
                print 'num fg avg: {}'.format(self._fg_num / self._count)
                print 'num bg avg: {}'.format(self._bg_num / self._count)
                print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        # sampled rois
        top[0].reshape(*top_all_rois.shape)
        top[0].data[...] = top_all_rois

        # classification labels
        top[1].reshape(*top_all_labels.shape)
        top[1].data[...] = top_all_labels

        # bbox_targets
        top[2].reshape(*top_all_bbox_targets.shape)
        top[2].data[...] = top_all_bbox_targets

        # bbox_inside_weights
        top[3].reshape(*top_all_bbox_inside_weights.shape)
        top[3].data[...] = top_all_bbox_inside_weights

        # bbox_outside_weights
        top[4].reshape(*top_all_bbox_inside_weights.shape)
        top[4].data[...] = np.array(top_all_bbox_inside_weights > 0).astype(np.float32)

        #print 'num fg: {} num bg: {}'.format((top_all_labels > 0).sum(), (top_all_labels == 0).sum())

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = int(start + 4)
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, flags, fg_rois_per_image, rois_per_image, num_classes, batch_index):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # filter gt_boxes
    pos_gt_num = np.count_nonzero(flags)
    neg_gt_num = len(flags) - pos_gt_num
    pos_gt_filter = np.where(flags.astype(np.bool))[0]
    neg_gt_filter = np.where(np.invert(flags.astype(np.bool)))[0]
    pos_gt_boxes = None
    neg_gt_boxes = None
    if pos_gt_num > 0:
        pos_gt_boxes = gt_boxes[pos_gt_filter]
    if neg_gt_num > 0:
        neg_gt_boxes = gt_boxes[neg_gt_filter]

    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    mask = np.in1d(gt_assignment, neg_gt_filter)
    labels[mask] = 0
    # set __background__ labels

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    #allen
    #print fg_inds
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    bg_inds = np.empty((0), dtype=np.int32)
    if neg_gt_num > 0:
        overlaps_with_false = bbox_overlaps(
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(neg_gt_boxes[:, :4], dtype=np.float))
        #argmax_overlaps_with_false = overlaps_with_false.argmax(axis=1)
        max_overlaps_with_false = overlaps_with_false.max(axis=1)
        bg_inds = np.where(max_overlaps_with_false > 0.15)[0]

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

    if len(bg_inds) < bg_rois_per_this_image:
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        ext_bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        rest_num = min(bg_rois_per_this_image - len(bg_inds), len(ext_bg_inds))
        ext_bg_inds = npr.choice(ext_bg_inds, size=rest_num, replace=False)
        bg_inds = np.concatenate((bg_inds, ext_bg_inds))
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)

    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    # test1 = rois
    # test2 = gt_boxes[gt_assignment[keep_inds], :4]
    # test3 = labels

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
