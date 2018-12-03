# -*- coding: utf-8 -*-
#!/usr/bin/env python
import _init_paths
import os, sys
os.environ['GLOG_minloglevel'] = '3'
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
from iouutils import IOU_multi, validBigBox
import numpy as np
import caffe
import cv2
CLASSES = ('__background__','hand')
#返回手势类别、坐标、分数，每行对应同一个手势
def demo(net, im, confTH, nmsTH):
    """Detect object classes in an image using pre-computed object proposals."""
    scores, boxes = im_detect(net, im)
    handcls = np.array([])
    handbox = np.array([])
    handscore = np.array([])
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1                                            # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]     # 300*4矩阵
        cls_scores = scores[:, cls_ind]                         # 300行
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nmsTH)
        dets = dets[keep, :]
        for i in xrange(dets.shape[0]):
            if (dets[i][4] > confTH):
                # nhand += 1
                handcls = np.append( handcls, CLASSES[cls_ind] )
                handbox = np.append( handbox, [ dets[i][0], dets[i][1], dets[i][2], dets[i][3] ] )
                handscore = np.append( handscore, [dets[i][4]] )
                cv2.rectangle(im, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), (0, 255, 0), 4)
                cv2.putText(im, str(dets[i][4]), (dets[i][0], dets[i][1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        handcls = np.reshape(handcls, (-1,1))
        handbox = np.reshape(handbox, (-1, 4))
        handscore = np.reshape(handscore, (-1,1))
        # print dets.shape[0], " handcls:", handcls, "handbox:\n", handbox, "score:", handscore
        return handcls, handbox, handscore
def get_gt(filename, shortSide):
    nhand_gt = int(filename.split(' ')[1])
    handcls_gt = np.array([])  # filename.split(' ')[2]
    box_gt = np.array([])
    box_scale = np.array([])
    for i in xrange(nhand_gt):
        handcls_gt = np.append(handcls_gt, filename.split(' ')[i * 5 + 2])
        box_gt = np.append(box_gt, np.array([int(filename.split(' ')[i * 5 + 3]), \
                                             int(filename.split(' ')[i * 5 + 4]), \
                                             int(filename.split(' ')[i * 5 + 5]), \
                                             int(filename.split(' ')[i * 5 + 6])]))
    handcls_gt = np.reshape(handcls_gt, (-1, 1))
    box_gt = np.reshape(box_gt, (-1, 4))
    for i in xrange(box_gt.shape[0]):
        bwidth = box_gt[i][2] - box_gt[i][0]
        bheight = box_gt[i][3] - box_gt[i][1]
        boxscale = shortSide/np.min([bwidth, bheight])
        print box_gt[i], boxscale, shortSide
        box_scale = np.append(box_scale, boxscale)
    box_scale = np.reshape(box_scale, (-1, 1))
    return nhand_gt, handcls_gt, box_gt, box_scale

