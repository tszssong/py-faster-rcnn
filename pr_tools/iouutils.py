# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import copy
import numpy.random as npr
import xml.etree.cElementTree as ET
from xml.etree.cElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import cv2
import numpy as np

def IOU(Reframe,GTframe):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]
    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1 + width2 - (endx - startx)
    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio
def AreaInGT(Reframe,GTframe):

    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]-Reframe[0]
    height1 = Reframe[3]-Reframe[1]
    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]-GTframe[0]
    height2 = GTframe[3]-GTframe[1]
    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1 + width2 - (endx - startx)
    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)
    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area2 = width2 * height2
        ratio = Area * 1. / (Area2 +  1)
    return ratio

def IOU_multi(box, boxes):
    maxIou = 0.0
    for box_idx in xrange(boxes.shape[0]):
        compBox = boxes[box_idx]
        iou = IOU(box, compBox)
        if iou > maxIou:
            maxIou = iou
#    print maxIou
    return maxIou

def containBox(box, boxes):
    contain = False
    for box_idx in xrange(boxes.shape[0]):
        compBox = boxes[box_idx]
        compW = compBox[2] - compBox[0]
        compH = compBox[3] - compBox[1]
        if( box[0]<compBox[0]-compW/10 and box[1]<compBox[1]-compH/10 and box[2]>compBox[2]+compW/10 and box[3]< compBox[3]+compH/10):
            contain = True
    return contain

def overlapSelf(Reframe,GTframe):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        ratio = Area * 1. / Area1
    return ratio

def overlapingOtherBox(crop_box, box_idx, f_boxes):
    overlap_flag = 0
    for otherbox_idx in xrange(f_boxes.shape[0]):
        if not box_idx == otherbox_idx:
            iou = IOU(crop_box, f_boxes[otherbox_idx])
            if iou > 0.01:
                overlap_flag = 1
    if overlap_flag == 1:
        return True

def validBigBox(boxes, shortSide, scale):
    isBig = 1
    for boxidx in xrange(boxes.shape[0]):
        if( boxes[boxidx][2] - boxes[boxidx][0] < float(shortSide)/scale) or ( boxes[boxidx][3] - boxes[boxidx][1] < float(shortSide)/scale) :
            isBig = 0
    return isBig 
#***************************gt re****************************#
def get_re_from_line(line):
    nhand_re = int(line.split(' ')[1])
    handcls_re = np.array([])  # line.split(' ')[2]
    box_re = np.array([])
    for i in xrange(nhand_re):
        handcls_re = np.append(handcls_re, line.split(' ')[i * 5 + 2])
        box_re = np.append(box_re, np.array([int(line.split(' ')[i * 5 + 3]), \
                                             int(line.split(' ')[i * 5 + 4]), \
                                             int(line.split(' ')[i * 5 + 5]), \
                                             int(line.split(' ')[i * 5 + 6])]))
    handcls_re = np.reshape(handcls_re, (-1, 1))
    box_re = np.reshape(box_re, (-1, 4))
    return nhand_re, handcls_re, box_re
def get_gt_from_xml(xml_path):
    tree = ET.parse(xml_path) 
    root = tree.getroot() 
    filename = root.find('filename').text

    nhand_in_gt = 0
    handcls_gt = np.array([], dtype=int)
    box_gt = np.array([], dtype=int)
    for object in root.findall('object'):  
        nhand_in_gt += 1

        hand_name = object.find('name').text
        handcls_gt = np.append(handcls_gt, hand_name)
        # print hand_name
        bndbox = object.find('bndbox')  
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        if xmin > xmax:
            tmp = xmin
            xmin = xmax
            xmax = tmp
        # print "xml x anno err: ", filename, xmin, xmax
        if ymin > ymax:
            tmp = ymin
            ymin = ymax
            ymax = tmp
        # print "xml y anno err: ", filename, ymin, ymax
        box_gt = np.append(box_gt, np.array([xmin, ymin, xmax, ymax]))
    handcls_gt = np.reshape( handcls_gt, (-1, 1) )
    box_gt = np.reshape( box_gt, (-1, 4) )
    return nhand_in_gt, handcls_gt, box_gt
    
def updateXMLbbox(xml_path, to_xml_path, addbox ):
    tree = ET.parse(xml_path) 
    root = tree.getroot() 
    node_object = Element('object')
    node_name = SubElement(node_object, 'name')
    node_name.text = 'hand'
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_difficult = SubElement(node_object, 'flag')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = str(int(addbox[0]))
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = str(int(addbox[1]))
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = str(int(addbox[2]))
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = str(int(addbox[3]))
    root.append(node_object)
    tree.write(os.path.join(to_xml_path, xml_path.split('/')[-1] ), encoding='utf-8')
    return True