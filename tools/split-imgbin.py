import sys
import os
import random
random.seed(888)

if len(sys.argv) < 5:
    print "Usage: python image_list img_root shuffle[0/1] prefix output_dir"
    exit(-1)

# im2bin path
IM2BIN = './im2bin'

fi = file(sys.argv[1])
lst = [line for line in fi]

img_root = sys.argv[2]

if sys.argv[3] == "1":
    random.shuffle(lst)

prefix = sys.argv[4]
output_dir = sys.argv[5]
if output_dir[-1] != '/':
    output_dir += '/'

fo = open("Makefile", "w")

objs = []
cmds = []
fw = None
sz = 0
img_cnt = 1;
cnt = 1

for item in lst:
    if sz + 10240 > (256<<20) or fw == None:
        lst_name = output_dir + prefix + ('-%08d' % cnt) + '.lst'
        bin_name = output_dir + prefix + ('-%08d' % cnt) + '.bin'
        objs.append(bin_name)
        if fw != None:
            fw.close()
        fw = open(lst_name, "w")
        cmd = "%s: %s\n\t%s %s %s %s" % (bin_name, lst_name,
                IM2BIN, lst_name, img_root, bin_name)
        cmds.append(cmd)
        sz = 0
        cnt += 1
        img_cnt = 1
    path = item.split('\t')[2][:-1]
    sz += os.path.getsize(img_root + path) + (img_cnt + 2) * 4
    fw.write(item)
    img_cnt += 1

obj = "all: " + ' '.join(objs) + '\n'
fo.write(obj)
fo.write('\n\n'.join(cmds))
fo.close()










