# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
import cv2

DEBUG = False

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(base_size=5, scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(cfg.TRAIN.IMS_PER_BATCH, 1, A * height, width)
        # bbox_targets
        top[1].reshape(cfg.TRAIN.IMS_PER_BATCH, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(cfg.TRAIN.IMS_PER_BATCH, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(cfg.TRAIN.IMS_PER_BATCH, A * 4, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        # test = bottom[0].data
        # assert bottom[0].data.shape[0] == 1, \
        #     'Only single item batches are supported'
        assert bottom[0].data.shape[0] == cfg.TRAIN.IMS_PER_BATCH
        num_images = bottom[0].data.shape[0]
        assert cfg.TRAIN.RPN_BATCHSIZE % num_images == 0
        batchsize_per_images = cfg.TRAIN.RPN_BATCHSIZE / num_images

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        # im_info
        im_info = bottom[2].data
        # flags
        flags = bottom[4].data

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        all_labels = np.empty((0, 1, A * height, width), dtype=np.float32)
        all_bbox_targets = np.empty((0, A * 4, height, width), dtype=np.float32)
        all_bbox_inside_weights = np.empty((0, A * 4, height, width), dtype=np.float32)
        all_bbox_outside_weights = np.empty((0, A * 4, height, width), dtype=np.float32)
        for batch_index in xrange(num_images):
            batch_gt_boxes_inds = np.where(gt_boxes[:, 5] == batch_index)[0]
            batch_gt_boxes = gt_boxes[batch_gt_boxes_inds, 0:-1]
            batch_flags_inds = np.where(flags[:, 1] == batch_index)[0]
            batch_flags = flags[batch_flags_inds, 0]
            assert np.all(batch_flags_inds == batch_gt_boxes_inds)
            # only keep anchors inside the image
            inds_inside = np.where(
                (all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < im_info[batch_index, 1] + self._allowed_border) &  # width
                (all_anchors[:, 3] < im_info[batch_index, 0] + self._allowed_border)    # height
            )[0]

            if DEBUG:
                print 'total_anchors', total_anchors
                print 'inds_inside', len(inds_inside)

            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]

            if DEBUG:
                print 'anchors.shape', anchors.shape

            # label: 1 is positive, 0 is negative, -1 is dont care
            labels = np.empty((len(inds_inside), ), dtype=np.float32)
            labels.fill(-1)

            # filter gt_boxes
            pos_gt_num = np.count_nonzero(batch_flags)
            neg_gt_num = len(batch_flags) - pos_gt_num
            pos_gt_filter = np.where(batch_flags.astype(np.bool))[0]
            neg_gt_filter = np.where(np.invert(batch_flags.astype(np.bool)))[0]
            pos_gt_boxes = None
            neg_gt_boxes = None
            if pos_gt_num > 0:
                pos_gt_boxes = batch_gt_boxes[pos_gt_filter]
            if neg_gt_num > 0:
                neg_gt_boxes = batch_gt_boxes[neg_gt_filter]

            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt)
            if pos_gt_num > 0:
                #print 'pos_gt_num > 0'
                overlaps_with_true = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(pos_gt_boxes, dtype=np.float))
                argmax_overlaps = overlaps_with_true.argmax(axis=1)
                max_overlaps = overlaps_with_true[np.arange(len(inds_inside)), argmax_overlaps]

                #test1 = sorted(max_overlaps, reverse=True)

                gt_argmax_overlaps = overlaps_with_true.argmax(axis=0)
                gt_max_overlaps = overlaps_with_true[gt_argmax_overlaps,
                                           np.arange(overlaps_with_true.shape[1])]
                gt_argmax_overlaps = np.where(overlaps_with_true == gt_max_overlaps)[0]

                if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                    # assign bg labels first so that positive labels can clobber them
                    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

                # fg label: for each gt, anchor with highest overlap
                labels[gt_argmax_overlaps] = 1

                # fg label: above threshold IOU
                labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

                if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                    # assign bg labels last so that negative labels can clobber positives
                    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

                #print '(-1)', np.sum(labels == -1), '(0)', np.sum(labels == 0), '(1)', np.sum(labels == 1), '(2)', np.sum(labels == 2)
            else:
                #print 'pos_gt_num <= 0'
                overlaps_with_true = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(batch_gt_boxes, dtype=np.float))
                argmax_overlaps = overlaps_with_true.argmax(axis=1)
                max_overlaps = overlaps_with_true[np.arange(len(inds_inside)), argmax_overlaps]
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            #print '-1', np.sum(labels == -1), '0', np.sum(labels == 0), '1', np.sum(labels == 1), '2', np.sum(labels == 2)

            if neg_gt_num > 0:
                #print 'neg_gt_num > 0'
                overlaps_with_false = bbox_overlaps(
                    np.ascontiguousarray(anchors, dtype=np.float),
                    np.ascontiguousarray(neg_gt_boxes, dtype=np.float))
                argmax_overlaps_with_false = overlaps_with_false.argmax(axis=1)
                max_overlaps_with_false = overlaps_with_false[np.arange(len(inds_inside)), argmax_overlaps_with_false]
                gt_argmax_overlaps = overlaps_with_false.argmax(axis=0)
                gt_max_overlaps = overlaps_with_false[gt_argmax_overlaps,
                                                     np.arange(overlaps_with_false.shape[1])]
                gt_argmax_overlaps = np.where(overlaps_with_false == gt_max_overlaps)[0]

                # fg label: for each gt, anchor with highest overlap
                labels[gt_argmax_overlaps] = 2

                # fg label: above threshold IOU
                labels[max_overlaps_with_false > 0.15] = 2

            #print '-1', np.sum(labels == -1), '0', np.sum(labels == 0), '1', np.sum(labels == 1), '2', np.sum(labels == 2)

            # subsample positive labels if we have too many
            num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * batchsize_per_images)
            fg_inds = np.where(labels == 1)[0]
            if len(fg_inds) > num_fg:
                disable_inds = npr.choice(
                    fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                labels[disable_inds] = -1

            # subsample negative labels if we have too many
            num_bg = batchsize_per_images - np.sum(labels == 1)
            if neg_gt_num > 0:
                #print 'neg_gt_num > 0 two'
                bg_inds = np.where(labels == 2)[0]
                bg_inds2 = np.where(labels == 0)[0]
                if len(bg_inds) > num_bg:
                    labels[bg_inds] = 0
                    labels[bg_inds2] = -1
                    disable_inds = npr.choice(
                        bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                    labels[disable_inds] = -1
                elif len(bg_inds) <= num_bg:
                    rest_bg = num_bg - len(bg_inds)
                    labels[bg_inds] = 0
                    labels[bg_inds2] = 0
                    disable_inds = npr.choice(bg_inds2, size=(len(bg_inds2) - rest_bg), replace=False)
                    labels[disable_inds] = -1
            else:
                #print 'neg_gt_num <= 0'
                bg_inds = np.where(labels == 0)[0]
                if len(bg_inds) > num_bg:
                    disable_inds = npr.choice(
                        bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                    labels[disable_inds] = -1
                    #print "was %s inds, disabling %s, now %s inds" % (
                        #len(bg_inds), len(disable_inds), np.sum(labels == 0))

            #print '(-1)', np.sum(labels == -1), '(0)', np.sum(labels == 0), '(1)', np.sum(labels == 1), '(2)', np.sum(labels == 2)

            bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
            bbox_targets = _compute_targets(anchors, batch_gt_boxes[argmax_overlaps, :])

            bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
            bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

            bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
            if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
                # uniform weighting of examples (given non-uniform sampling)
                num_examples = np.sum(labels >= 0)
                positive_weights = np.ones((1, 4)) * 1.0 / num_examples
                negative_weights = np.ones((1, 4)) * 1.0 / num_examples
            else:
                assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                        (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
                positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                    np.sum(labels == 1))
                negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                    np.sum(labels == 0))
            bbox_outside_weights[labels == 1, :] = positive_weights
            bbox_outside_weights[labels == 0, :] = negative_weights

            if DEBUG:
                self._sums += bbox_targets[labels == 1, :].sum(axis=0)
                self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
                self._counts += np.sum(labels == 1)
                means = self._sums / self._counts
                stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
                print 'means:'
                print means
                print 'stdevs:'
                print stds

            # for i in xrange(len(batch_gt_boxes)):
            #     [x1, y1, x2, y2] = batch_gt_boxes[i, :-1].astype(np.int32)
            #     cv2.rectangle(cfg.TRAIN.IMAGES_LIST[batch_index], (x1, y1), (x2, y2), (0, 255, 0), 5)
            # false_anchors = anchors[np.where(labels == 0)[0]]
            # for i in xrange(len(false_anchors)):
            #     [x1, y1, x2, y2] = false_anchors[i].astype(np.int32)
            #     cv2.rectangle(cfg.TRAIN.IMAGES_LIST[batch_index], (x1, y1), (x2, y2), (0, 255, 255), 2)
            # true_anchors = anchors[np.where(labels == 1)[0]]
            # for i in xrange(len(true_anchors)):
            #     [x1, y1, x2, y2] = true_anchors[i].astype(np.int32)
            #     cv2.rectangle(cfg.TRAIN.IMAGES_LIST[batch_index], (x1, y1), (x2, y2), (255, 0, 255), 2)
            # cv2.imshow('test_im', cfg.TRAIN.IMAGES_LIST[batch_index])
            # cv2.waitKey()

            # map up to original set of anchors
            labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
            bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
            bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
            bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

            if DEBUG:
                print 'rpn: max max_overlap', np.max(max_overlaps)
                print 'rpn: num_positive', np.sum(labels == 1)
                print 'rpn: num_negative', np.sum(labels == 0)
                self._fg_sum += np.sum(labels == 1)
                self._bg_sum += np.sum(labels == 0)
                self._count += 1
                print 'rpn: num_positive avg', self._fg_sum / self._count
                print 'rpn: num_negative avg', self._bg_sum / self._count

            # labels
            labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
            labels = labels.reshape((1, 1, A * height, width))
            all_labels = np.vstack((all_labels, labels))

            # bbox_targets
            bbox_targets = bbox_targets \
                .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
            all_bbox_targets = np.vstack((all_bbox_targets, bbox_targets))

            # bbox_inside_weights
            bbox_inside_weights = bbox_inside_weights \
                .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
            assert bbox_inside_weights.shape[2] == height
            assert bbox_inside_weights.shape[3] == width
            all_bbox_inside_weights = np.vstack((all_bbox_inside_weights, bbox_inside_weights))

            # bbox_outside_weights
            bbox_outside_weights = bbox_outside_weights \
                .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
            assert bbox_outside_weights.shape[2] == height
            assert bbox_outside_weights.shape[3] == width
            all_bbox_outside_weights = np.vstack((all_bbox_outside_weights, bbox_outside_weights))

        top[0].reshape(*all_labels.shape)
        top[0].data[...] = all_labels

        top[1].reshape(*all_bbox_targets.shape)
        top[1].data[...] = all_bbox_targets

        top[2].reshape(*all_bbox_inside_weights.shape)
        top[2].data[...] = all_bbox_inside_weights

        top[3].reshape(*all_bbox_outside_weights.shape)
        top[3].data[...] = all_bbox_outside_weights

        #print 'rpn: num_positive', np.sum(all_labels == 1), 'rpn: num_negative', np.sum(all_labels == 0)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
