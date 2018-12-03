# -*- coding: utf-8 -*-
#!/usr/bin/env python
import _init_paths
import os, sys
os.environ['GLOG_minloglevel'] = '3'
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.test import vis_detections
from fast_rcnn.nms_wrapper import nms
from iouutils import IOU_multi, get_gt_from_xml
import numpy as np
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
CLASSES = ('__background__','hand')
iouP = 0.3
CONF_THRESH = 0.9
NMS_THRESH = 0.01
iterStart = int( sys.argv[1] )
iterEnd = int( sys.argv[2] )
iterStep = int( sys.argv[3] )
fromDir =  "../data/test/"
from_img_path = fromDir + "gzTest5ali-img/"
from_xml_path = fromDir + "gzTest5ali-xml/"
readlists = "gzTest5ali.txt"
#from_img_path = fromDir + "gzTest180-img/"
#from_xml_path = fromDir + "gzTest180-xml/"
#readlists = "gzTest1116.txt"
toTxtDir = "../output/gzTestAli/modelsTxt/"
if not os.path.isdir(toTxtDir):
    os.makedirs(toTxtDir)
readFile = open(fromDir + readlists, "r")
prototxt = "../models/pascal_voc/MMCV5BNS16/faster_rcnn_end2end/test.prototxt"
modelPath = "../output/faster_rcnn_end2end/voc_2007_trainval/"
modelprefix = "3iter2_f327w_dq1_iter_"
def speical_config():
    caffe.set_mode_gpu()
    caffe.set_device(2)
    cfg.GPU_ID =  2
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALES = [180,]
    cfg.TEST.MAX_SIZE = 320
    cfg.DEDUP_BOXES = 1./16.
    cfg.TEST.USE_RPN = False
    cfg.TEST.RPN_PRE_NMS_TOP_N = 50
    cfg.TEST.RPN_POST_NMS_TOP_N = 10

#返回手势类别、坐标、分数，每行对应同一个手势
def demo(net, im):
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
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        for i in xrange(dets.shape[0]):
            if (dets[i][4] > CONF_THRESH):
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
def get_gt(filename):

    nhand_gt = int(filename.split(' ')[1])

    handcls_gt = np.array([])  # filename.split(' ')[2]

    box_gt = np.array([])
    for i in xrange(nhand_gt):
        handcls_gt = np.append(handcls_gt, filename.split(' ')[i * 5 + 2])
        box_gt = np.append(box_gt, np.array([int(filename.split(' ')[i * 5 + 3]), \
                                             int(filename.split(' ')[i * 5 + 4]), \
                                             int(filename.split(' ')[i * 5 + 5]), \
                                             int(filename.split(' ')[i * 5 + 6])]))
    handcls_gt = np.reshape(handcls_gt, (-1, 1))
    box_gt = np.reshape(box_gt, (-1, 4))
    return nhand_gt, handcls_gt, box_gt

if __name__ == '__main__':
    speical_config()
    print "test", modelprefix, "from%dw"%iterStart, "to%dw"%iterEnd, "with step=%dw"%iterStep, "and TH=%.2f, IOU=%.2f"%(CONF_THRESH, iouP)
    filelists = readFile.readlines()
    recallSet = np.array([], dtype=np.float32)
    precSet = np.array([], dtype=np.float32)
    totalResultsFile = open(toTxtDir + "/totalRet_"+ modelprefix + str(iterStart) + '_' + str(iterEnd) +".txt", "w")
    totalResultsFile.write("\ntesting "+modelprefix + "s with iou= " + str(iouP) + "CONF_THRESH" + str())
    totalResultsFile.write( "\npre nms:"+ str(cfg.TEST.RPN_PRE_NMS_TOP_N) )
    totalResultsFile.write( "\npost nms:"+ str(cfg.TEST.RPN_POST_NMS_TOP_N) )
    totalResultsFile.write("\ntest lists:"+readlists )
    for model_idx in xrange(iterStart, iterEnd, iterStep):
        caffemodel = modelprefix +str(model_idx)+"0000.caffemodel"
        writeFile = open(toTxtDir + "/ret_"+ caffemodel.split('.')[0]+".txt", "w")
        net = caffe.Net(prototxt, modelPath+caffemodel, caffe.TEST)
        numFrame = 0
        n_pos_gt = 0
        n_pos_re = 0
        n_pos_tp = 0
        for filename in filelists:
            picName = filename.strip().split(' ')[0].split('.')[0]+'.jpg'
            nhand_gt, handcls_gt, box_gt = get_gt_from_xml( from_xml_path + filename.strip().split(' ')[0].split('.')[0] + '.xml' )
            
            n_pos_gt += nhand_gt
            frame = cv2.imread(from_img_path+picName)
            numFrame+=1
            handcls, handbox, handscore = demo(net, frame)
            nhand = handbox.shape[0]
            n_pos_re += nhand
            nTP = 0
            if nhand > 0:
                writeFile.write(picName + ' ' + str(nhand))
                for i in xrange(0, nhand):
                    writeFile.write(' hand %d %d %d %d'%(handbox[i][0],handbox[i][1],handbox[i][2],handbox[i][3]))
                    if ( IOU_multi(handbox[i],box_gt) > iouP):
                        nTP += 1
                writeFile.write("\n")
            n_pos_tp += nTP
        modelrecall = float(n_pos_tp) / float(n_pos_gt)
        recallSet=np.append(recallSet, modelrecall)
        modelprec = float(n_pos_tp) / float(n_pos_re)
        precSet = np.append(precSet, modelprec)
        print "model%3dw: gt =%4d, re =%4d, tp =%4d, prec=%.4f, recall=%.4f"%(model_idx, n_pos_gt,n_pos_re,  n_pos_tp, \
                                                                    float(n_pos_tp) / float(n_pos_re), float(n_pos_tp) / float(n_pos_gt))
        totalResultsFile.write( "\nmodel%3dw: gt =%4d, re =%4d, tp =%4d, prec=%.4f, recall=%.4f"%(model_idx, n_pos_gt,n_pos_re,  n_pos_tp, \
                                                                    float(n_pos_tp) / float(n_pos_re), float(n_pos_tp) / float(n_pos_gt)) )
        writeFile.close()
    print np.min(precSet), np.where(precSet==min(precSet))
    print np.max(recallSet), np.where(recallSet==max(recallSet))
    print np.mean(recallSet)
    totalResultsFile.write( "\nbest recall: %.4f"%np.max(recallSet) )
    maxlist = np.where( recallSet==max(recallSet) )
    for i in xrange(maxlist[0].shape[0]):
        totalResultsFile.write("\nbest recall idx: %d" % maxlist[0][i])
    readFile.close()
