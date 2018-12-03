import sys
import os
import shutil

fileList=sys.argv[1]
fromDir = 'JPEGImages/'
targetDir=fileList.split('.')[0]+'-img'

if not os.path.isdir("./"+targetDir+"/"):
    os.makedirs("./"+targetDir+"/")

f = open(fileList)
line = f.readline()

while line:
    a = line.strip('\n')
    filename =  a.split('.')[0]
    exists = os.path.exists(os.getcwd()+'/'+fromDir+'/'+filename+'.jpg')
    if exists :
        shutil.copy(fromDir+'/'+filename+'.jpg',targetDir+'/'+filename+'.jpg')
    else:
        print filename +' not exists'
    line = f.readline()
f.close()

fromDir = 'Annotations/'
targetDir=fileList.split('.')[0]+'-xml'

if not os.path.isdir("./"+targetDir+"/"):
    os.makedirs("./"+targetDir+"/")

f = open(fileList)
line = f.readline()

while line:
    a = line.strip('\n')
    filename =  a.split('.')[0]
    #    print filename
    exists = os.path.exists(os.getcwd()+'/'+fromDir+'/'+filename+'.xml')
    if exists :
        shutil.copy(fromDir+'/'+filename+'.xml',targetDir+'/'+filename+'.xml')
    else:
        print filename +' not exists'
    line = f.readline()
f.close()



