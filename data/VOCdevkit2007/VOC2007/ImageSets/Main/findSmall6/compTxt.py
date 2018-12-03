import sys
import os

refilepath = sys.argv[1]
gtfilepath = sys.argv[2]

relist = []
for line in open(refilepath):
    filename = line.strip().split("/")[-1]
    print "re:", filename
    relist.append(filename)
print relist
gtlist = []
gtdict = {}
for line in open(gtfilepath):
    filename = line.strip().split("/")[-1]
#    print "gt:",filename
    gtlist.append(filename)
    gtdict[filename] = filename
#print gtlist

f = open('1in2'+sys.argv[1], 'w')
for filename in relist:
    if gtdict.has_key(filename):
        f.write(gtdict[filename]+'\n')
    else:
        print filename
f.close()

f = open('2notin1'+sys.argv[1], 'w')
for filename in gtlist:
    if not filename in relist:
        f.write(filename+'\n')
    else:
        continue
f.close()



