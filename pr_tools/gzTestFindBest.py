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
from fr_cnn_utils import demo
import numpy as np
caffe_root = './caffe-fast-rcnn/'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
SAVE_PIC = True
CLASSES = ('__background__','hand')
iouP = 0.4
CONF_THRESH = 0.9
NMS_THRESH = 0.01
iterStart = int( sys.argv[1] )
iterEnd = int( sys.argv[2] )
iterStep = int( sys.argv[3] )
prototxt = "../models/pascal_voc/MMCV5BNS16/faster_rcnn_end2end/test.prototxt"
modelPath = "../output/faster_rcnn_end2end/voc_2007_trainval/"
modelprefix = "iter3on525w_iter_"
fromDir =  "../data/test/"
# from_img_path = fromDir + "JPEGImages/"
# from_xml_path = fromDir + "Annotations/"
from_img_path = fromDir + "gzTest180-img/"
from_xml_path = fromDir + "gzTest180-xml/"
readlists = "gzTest1116.txt"
toTxtDir = "../output/gzTest1116/modelsTxt/"
if not os.path.isdir(toTxtDir):
    os.makedirs(toTxtDir)
if SAVE_PIC:
    toDir = "../output/gzTest1116/" + modelprefix+sys.argv[1] +'_'+str(CONF_THRESH)+"_"+str(iouP)+"/"
    if not os.path.isdir(toDir):
        os.makedirs(toDir)
    if not os.path.isdir(toDir+"_R/"):
        os.makedirs(toDir+"_R/")
    if not os.path.isdir(toDir+"_U/"):
        os.makedirs(toDir+"_U/")
readFile = open(fromDir + readlists, "r")
def speical_config():
    caffe.set_mode_cpu()
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALES = [180,]
    cfg.TEST.MAX_SIZE = 320
    cfg.DEDUP_BOXES = 1./16.
    cfg.TEST.USE_RPN = False
    cfg.TEST.RPN_PRE_NMS_TOP_N = 50
    cfg.TEST.RPN_POST_NMS_TOP_N = 10

if __name__ == '__main__':
    speical_config()
    print "test", modelprefix, "from%dw"%iterStart, "to%dw"%iterEnd, "with step=%dw"%iterStep, "and TH=%.2f, IOU=%.2f"%(CONF_THRESH, iouP)
    filelists = readFile.readlines()
    recallSet = np.array([], dtype=np.float32)
    precSet = np.array([], dtype=np.float32)
    totalResultsFile = open(toTxtDir + "/totalRet_"+ modelprefix + str(iterStart) + '_' + str(iterEnd) +".txt", "w")
    totalResultsFile.write("use %s : iou=%.2f, thresh=%.2f, preNMS=%d, posNms=%d"%(readlists, iouP, CONF_THRESH, \
                            cfg.TEST.RPN_PRE_NMS_TOP_N, cfg.TEST.RPN_POST_NMS_TOP_N) )
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
            handcls, handbox, handscore = demo(net, frame, CONF_THRESH, NMS_THRESH)
            nhand = handbox.shape[0]
            n_pos_re += nhand
            nTP = 0
            if nhand > 0:
                writeFile.write(picName + ' ' + str(nhand))
                for i in xrange(0, nhand):
                    writeFile.write(' hand %d %d %d %d'%(handbox[i][0],handbox[i][1],handbox[i][2],handbox[i][3]))
                    if ( IOU_multi(handbox[i],box_gt) > iouP):
                        nTP += 1
                    else:
                        print picName
                writeFile.write("\n")
                if SAVE_PIC:
                    cv2.imwrite(toDir+"/_R/" + picName, frame)
            else:
                cv2.imwrite(toDir+"/_U/" + picName, frame)
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
    print np.max(recallSet), np.where(recallSet==max(recallSet))
    print np.mean(recallSet)
    totalResultsFile.write( "\nbest recall: %.4f"%np.max(recallSet) )
    maxlist = np.where( recallSet==max(recallSet) )
    for i in xrange(maxlist[0].shape[0]):
        totalResultsFile.write("\nbest recall idx: %d" % maxlist[0][i])
    readFile.close()