# -*- coding: utf-8 -*-
#!/usr/bin/env python
import _init_paths
import os, sys
os.environ['GLOG_minloglevel'] = '3'
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
from iouutils import IOU_multi, get_gt_from_xml, updateXMLbbox
from fr_cnn_utils import demo
import numpy as np
import copy
SAVE_PIC = True
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
CLASSES = ('__background__','hand')
iouS = 0.1
iouM = 0.3
iouB = 0.5
CONF_THRESH = 0.9
NMS_THRESH = 0.01
prototxt = "../models/pascal_voc/MMCV5BNS16/faster_rcnn_end2end/test.prototxt"
model_path = "../output/faster_rcnn_end2end/voc_2007_trainval/"
caffemodel = "iter3on525w_iter_3270000.caffemodel"
from_img_path = "../data/VOCdevkit2007/VOC2007/JPEGImages/"
from_xml_path = "../data/VOCdevkit2007/VOC2007/Annotations/"
testlists = "trainval.txt"
toDir = "../output/unRecalled/" + caffemodel.split('.')[0]+testlists.split('.')[0]+"/"
if not os.path.isdir(toDir):
    os.makedirs(toDir)
if SAVE_PIC:
    if not os.path.isdir(toDir+"_S/"):
        os.makedirs(toDir+"_S/")
    if not os.path.isdir(toDir+"_B/"):
        os.makedirs(toDir+"_B/")
    if not os.path.isdir(toDir+"_U/"):
        os.makedirs(toDir+"_U/")


readFile = open("../data/" +testlists, "r")
def speical_config():
    caffe.set_mode_gpu()
    caffe.set_device(2)
    cfg.GPU_ID =  2
    cfg.TEST.HAS_RPN = True         # Use RPN for proposals
    cfg.TEST.SCALES = [180,]
    cfg.TEST.MAX_SIZE = 320
    cfg.DEDUP_BOXES = 1./16.
    cfg.TEST.USE_RPN = False
    cfg.TEST.RPN_PRE_NMS_TOP_N = 50
    cfg.TEST.RPN_POST_NMS_TOP_N = 10

if __name__ == '__main__':
    speical_config() 
    filelists = readFile.readlines()
    write_unrecall = open(toDir+"../unRecall_"+ caffemodel.split('.')[-2].split('/')[-1]+".txt", "w")
    write_iouS = open(toDir+"../iouS_"+ caffemodel.split('.')[-2].split('/')[-1]+".txt", "w")
    write_iouB = open(toDir+"../iouB_"+ caffemodel.split('.')[-2].split('/')[-1]+".txt", "w")
    write_small = open(toDir+"../smallGT_"+ caffemodel.split('.')[-2].split('/')[-1]+".txt", "w")
    net = caffe.Net(prototxt, model_path+caffemodel, caffe.TEST)
    # print '\n\nLoaded network: {:s}'.format(caffemodel)
    numFrame = 0
    n_hand_gt = 0
    handcls_gt = np.array([], dtype = int)
    box_gt = np.array([], dtype = int)
    for line in filelists:
        fileame = line.strip().split(' ')[0].split('.')[0]
        xml_path = from_xml_path + fileame+ '.xml'
        frame = cv2.imread(from_img_path+fileame+'.jpg')
        try:
            frame.shape
        except:
            print fileame
            continue
        n_hand_gt, handcls_gt, box_gt = get_gt_from_xml( from_xml_path + fileame+ '.xml' )
        small_flag = False
        for gt_idx in xrange(box_gt.shape[0]):
            if min(box_gt[gt_idx][3]-box_gt[gt_idx][1], box_gt[gt_idx][2]-box_gt[gt_idx][0]) < min(frame.shape[0], frame.shape[1])/6.:
                # print box_gt[gt_idx], frame.shape
                small_flag = True
        if(small_flag): #small box in gt: no need to recall
            write_small.write(fileame + "\n") 
            continue
        copyframe = frame.copy()
        numFrame+=1
        handcls, handbox, handscore = demo(net, frame, CONF_THRESH, NMS_THRESH)
        nhand = handbox.shape[0]
        nTP = 0
        if nhand > 0:           #recalled: find iou unright
            for i in xrange(0, nhand):
                iou = IOU_multi(handbox[i], box_gt)
                if( iou <= iouM and iou> iouS):
                    cv2.imwrite(toDir+"_S/" + fileame + '.jpg' , frame)
                    write_iouS.write(fileame + "\n") 
                elif( iou <= iouB and iou> iouM):
                    cv2.imwrite(toDir+"_B/" + fileame + '.jpg' , frame)
                    write_iouB.write(fileame + "\n") 
        else:                   #unrecalled: savelists
            write_unrecall.write(fileame + "\n")
            cv2.imwrite(toDir+"_U/" + fileame + '.jpg', frame)
    if numFrame%100 == 0:
        print numFrame
    print "%6d negInNeg and %6d negInNull in %6d frame"%(n_false_neg, n_false_null, numFrame)
