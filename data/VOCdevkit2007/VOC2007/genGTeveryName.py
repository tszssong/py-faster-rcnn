# -*- coding: utf-8 -*-
import sys
import os
import re
sys.path.append("/usr/local/Cellar/opencv/3.4.1_2/lib/python2.7/site-packages")
import cv2
import random
import xml.etree.cElementTree as ET

gtName = sys.argv[1]+'.txt'
xml_path = sys.argv[1]

m = open(gtName, "w")
for dirpath,dirnames,filenames in os.walk(xml_path):
    for f in filenames:
        if not '.xml' in f:
            continue
        print f
        tree = ET.parse(xml_path+"/"+f)  # 打开xml文档
        root = tree.getroot()  # 获得root节点
        filename = root.find('filename').text
        num = 0
        objNameList = []
        for object in root.findall('object'):
            num += 1
            name = object.find('name').text  # 子节点下节点name的值
            print name
            if not name in objNameList:
                objNameList.append(name)

        if num == 0 or len(objNameList)>1:
            continue                         #跳过一张图中多种手
        else:
            m.write(filename + ' ' + str(num) + ' ' )

        for object in root.findall('object'):  # 找到root节点下的所有object节点
            m.write( objNameList[0] +' ')
            bndbox = object.find('bndbox')  # 子节点下属性bndbox的值
            xmin = bndbox.find('xmin').text
            ymin = bndbox.find('ymin').text
            xmax = bndbox.find('xmax').text
            ymax = bndbox.find('ymax').text
            if int(xmin)<=int(xmax) and int(ymin)<=int(ymax):
                m.write(xmin +' ' + ymin +' ' + xmax + ' ' + ymax + ' ')
            elif int(xmin)>int(xmax) and int(ymin)<=int(ymax):
                m.write(xmax +' ' + ymin +' ' + xmin + ' ' + ymax + ' ')
            elif int(xmin)<int(xmax) and int(ymin)>=int(ymax):
                m.write(xmin +' ' + ymax +' ' + xmax + ' ' + ymin + ' ')
            else:
                m.write(xmax +' ' + ymax +' ' + xmin + ' ' + ymin + ' ')
        print objNameList

        m.write('\n')
m.close()


