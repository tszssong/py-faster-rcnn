# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import random

DEBUG = False

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales, im_shapes, rbbs = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    if cfg.TRAIN.HAS_RPN:
        #assert len(im_scales) == 1, "Single batch only"
        #assert len(roidb) == 1, "Single batch only"
        # gt boxes: (x1, y1, x2, y2, cls, inds)
        gt_boxes = np.empty((0, 6), dtype=np.float32)
        # flags: (flag, inds)
        flags = np.empty((0, 2), dtype=np.int32)
        # im_info: (h, w, scale)
        im_infos = np.empty((0, 3), dtype=np.float32)
        for i in xrange(num_images):
            gt_inds = np.where(roidb[i]['gt_classes'] != 0)[0]
            single_gt_boxes = np.empty((len(gt_inds), 6), dtype=np.float32)
            if len(rbbs[i]) > 0:
                single_gt_boxes[:, 0:4] = rbbs[i][gt_inds, :] * im_scales[i]
            else:
                single_gt_boxes[:, 0:4] = roidb[i]['boxes'][gt_inds, :] * im_scales[i]
            single_gt_boxes[:, 4] = roidb[i]['gt_classes'][gt_inds]
            single_gt_boxes[:, 5] = i
            gt_boxes = np.vstack((gt_boxes, single_gt_boxes))

            single_flags = np.empty((len(gt_inds), 2), dtype=np.int32)
            single_flags[:, 0] = roidb[i]['flags'][gt_inds]
            single_flags[:, 1] = i
            flags = np.vstack((flags, single_flags))

            single_im_info = np.array([im_shapes[i, 0], im_shapes[i, 1], im_scales[i]])
            im_infos = np.vstack((im_infos, single_im_info))
        blobs['gt_boxes'] = gt_boxes
        blobs['flags'] = flags
        blobs['im_info'] = im_infos
        # gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
        # gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        # gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        # gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        # blobs['flags'] = roidb[0]['flags'][gt_inds]
        # blobs['gt_boxes'] = gt_boxes
        # blobs['im_info'] = np.array(
        #     [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        #     dtype=np.float32)
    else: # not using RPN
        # Now, build the region of interest and label blobs
        rois_blob = np.zeros((0, 5), dtype=np.float32)
        labels_blob = np.zeros((0), dtype=np.float32)
        bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
        bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
        # all_overlaps = []
        for im_i in xrange(num_images):
            labels, overlaps, im_rois, bbox_targets, bbox_inside_weights \
                = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                               num_classes)

            # Add to RoIs blob
            rois = _project_im_rois(im_rois, im_scales[im_i])
            batch_ind = im_i * np.ones((rois.shape[0], 1))
            rois_blob_this_image = np.hstack((batch_ind, rois))
            rois_blob = np.vstack((rois_blob, rois_blob_this_image))

            # Add to labels, bbox targets, and bbox loss blobs
            labels_blob = np.hstack((labels_blob, labels))
            bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
            bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
            # all_overlaps = np.hstack((all_overlaps, overlaps))

        # For debug visualizations
        # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

        blobs['rois'] = rois_blob
        blobs['labels'] = labels_blob

        if cfg.TRAIN.BBOX_REG:
            blobs['bbox_targets'] = bbox_targets_blob
            blobs['bbox_inside_weights'] = bbox_inside_blob
            blobs['bbox_outside_weights'] = \
                np.array(bbox_inside_blob > 0).astype(np.float32)

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)

    return labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    rbbs = []
    cfg.TRAIN.IMAGES_LIST = []
    for i in xrange(num_images):
        #print roidb[i]['image']
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        use_blur = False
        use_lighting = False
        use_rotate = False
        use_bgr_offset = False
        if cfg.TRAIN.USE_BLUR and random.random() < cfg.TRAIN.BLUR_PERCENT:
            use_blur = True
        if cfg.TRAIN.USE_LIGHTING and random.random() < cfg.TRAIN.LIGHTING_PERCENT:
            use_lighting = True
        if cfg.TRAIN.USE_ROTATE and random.random() < cfg.TRAIN.ROTATE_PERCENT:
            use_rotate = True
        if cfg.TRAIN.USE_BGR_OFFSET and random.random() < cfg.TRAIN.BGR_OFFSET_PERCENT:
            use_bgr_offset = True

        if use_rotate:
            ori_im = np.copy(im)
            im_height, im_width = im.shape[:2]
            rot_d = np.random.randint(-cfg.TRAIN.ROTATE_DEGREE, cfg.TRAIN.ROTATE_DEGREE)
            M = cv2.getRotationMatrix2D((im_width / 2, im_height / 2), rot_d, 1)
            im = cv2.warpAffine(im, M, (im_width, im_height), None, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE)
            rotate_boxes = np.empty((0, 4), dtype=np.float32)
            for j in xrange(len(roidb[i]['boxes'])):
                [x1, y1, x2, y2] = [int(x) for x in roidb[i]['boxes'][j, :]]
                new_pt1 = np.dot(M, np.array([x1, y1, 1]).transpose()).astype(np.int32).transpose()
                new_pt2 = np.dot(M, np.array([x2, y2, 1]).transpose()).astype(np.int32).transpose()
                new_pt3 = np.dot(M, np.array([x1, y2, 1]).transpose()).astype(np.int32).transpose()
                new_pt4 = np.dot(M, np.array([x2, y1, 1]).transpose()).astype(np.int32).transpose()
                rect_pts = np.array([[new_pt1, new_pt2, new_pt3, new_pt4]])
                x, y, w, h = cv2.boundingRect(rect_pts)
                offset_x = 0
                offset_y = 0
                if x < 0:
                    offset_x = -x
                    x = 0
                if y < 0:
                    offset_y = -y
                    y = 0
                if x + w - offset_x < 0 or y + h - offset_y < 0:
                    rotate_boxes = []
                    im = ori_im
                    break
                if x + w - offset_x > im_width:
                    w = im_width - 1 - x - offset_x
                if y + h - offset_y > im_height:
                    h = im_height - 1 - y - offset_y
                rotate_boxes = np.vstack((rotate_boxes, np.array([x, y, x + w, y + h])))
            rbbs.append(rotate_boxes)
            if DEBUG:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow('rotate', im)
        else:
            rbbs.append([])

        if use_blur:
            kernel_size = np.random.randint(int(min(im.shape[0], im.shape[1]) * 0.03))
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1
            im = cv2.GaussianBlur(im, (kernel_size, kernel_size), 0)
            if DEBUG:
                cv2.imshow('blur_img', im)

        if use_lighting:
            im = im.astype(np.uint32) + np.random.randint(-100, 100)
            im[im > 255] = 255
            im[im < 0] = 0
            im = im.astype(np.uint8)
            if DEBUG:
                cv2.imshow('lighting_im', im)

        if use_bgr_offset:
            im = im.astype(np.int64)
            im[:, :, 0] += np.random.randint(-20, 20)
            im[:, :, 1] += np.random.randint(-20, 20)
            im[:, :, 2] += np.random.randint(-20, 20)
            im[im > 255] = 255
            im[im < 0] = 0
            im = im.astype(np.uint8)
            if DEBUG:
                cv2.imshow('bgr_offset_im', im)

        if DEBUG and (use_lighting or use_rotate or use_blur or use_bgr_offset):
            cv2.waitKey()
            cv2.destroyAllWindows()

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob, im_shapes = im_list_to_blob(processed_ims)

    return blob, im_scales, im_shapes, rbbs

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
