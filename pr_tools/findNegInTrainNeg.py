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
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
CLASSES = ('__background__','hand')
iouN = 0.05
CONF_THRESH = 0.9
NMS_THRESH = 0.01
prototxt = "../models/pascal_voc/MMCV5BNS16/faster_rcnn_end2end/test.prototxt"
model_path = "../output/faster_rcnn_end2end/voc_2007_trainval/"
caffemodel = "iter3on525w_iter_3270000.caffemodel"
from_img_path = "../data/VOCdevkit2007/VOC2007/JPEGImages/"
from_xml_path = "../data/VOCdevkit2007/VOC2007/Annotations/"
to_img_path = "../output/negInTrain"+caffemodel.split('.')[0].split('_')[-1]+"-img/"
to_xml_path = "../output/negInTrain"+caffemodel.split('.')[0].split('_')[-1]+"-xml/"
if not os.path.isdir(to_img_path):
    os.makedirs(to_img_path)
if not os.path.isdir(to_xml_path):
    os.makedirs(to_xml_path)

testlists = "negTest.txt"
readFile = open("../data/neg/" +testlists, "r")
def speical_config():
    caffe.set_mode_cpu()
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
    writeFile = open(to_xml_path+"../neg_"+ caffemodel.split('.')[-2].split('/')[-1]+".txt", "w")
    net = caffe.Net(prototxt, model_path+caffemodel, caffe.TEST)
    # print '\n\nLoaded network: {:s}'.format(caffemodel)
    numFrame = 0
    n_hand_gt = 0
    n_false_null = 0
    n_false_neg = 0
    handcls_gt = np.array([], dtype = int)
    box_gt = np.array([], dtype = int)
    for line in filelists:
        fileame = line.strip().split(' ')[0].split('.')[0]
        xml_path = from_xml_path + fileame+ '.xml'
        n_hand_gt, handcls_gt, box_gt = get_gt_from_xml( from_xml_path + fileame+ '.xml' )
        frame = cv2.imread(from_img_path+fileame+'.jpg')
        copyframe = frame.copy()
        numFrame+=1
        handcls, handbox, handscore = demo(net, frame, CONF_THRESH, NMS_THRESH)
        nhand = handbox.shape[0]
        nTP = 0
        if nhand > 0:
            for i in xrange(0, nhand):
                writeFile.write(' hand %d %d %d %d'%(handbox[i][0],handbox[i][1],handbox[i][2],handbox[i][3]))
                if (IOU_multi(handbox[i], box_gt) > iouN):
                    n_false_neg += 1
                else:
                    for gt_idx in xrange(box_gt.shape[0]):
                        print box_gt[gt_idx]
                        cv2.rectangle(frame, (box_gt[gt_idx][0], box_gt[gt_idx][1]), (box_gt[gt_idx][2], box_gt[gt_idx][3]), (255, 0, 0), 4)
                    n_false_null += 1
                updateXMLbbox(xml_path, to_xml_path, handbox[i] )
                cv2.imwrite(to_img_path+fileame+'.jpg', frame)
                writeFile.write(fileame + '.jpg ' + str(nhand))          
            writeFile.write("\n")
    print "%6d negInNeg and %6d negInNull in %6d frame"%(n_false_neg, n_false_null, numFrame)