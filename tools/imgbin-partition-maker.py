import sys
import os
import random
import argparse

random.seed(888)


parser = argparse.ArgumentParser(description='Generate a Makfile to make partition imgbin file for cxxnet')
parser.add_argument('--img_list', required=True, help="path to list of all images")
parser.add_argument('--img_root', required=True, help="prefix path to the file path in img_list")
parser.add_argument('--im2rec', default='../bin/im2rec', help="path to im2rec tools")
parser.add_argument('--partition_size', default="256", help="max size of single bin file")
parser.add_argument('--shuffle', default='0', help="Shuffle the list or not")
parser.add_argument('--prefix', required=True, help="Prefix of output image lists and bins")
parser.add_argument('--out', required=True, help="Output folder for image bins and lists")
parser.add_argument('--resize', required=True, help="New size of image (-1 for do nothing)")
parser.add_argument('--makefile', default="Gen.mk", help="name of generated Makefile")


args = parser.parse_args()
# im2bin path
IM2BIN = args.im2rec

new_size = "resize=" + args.new_size

fi = file(args.img_list)
lst = [line for line in fi]

img_root = args.img_root

if args.shuffle == "1":
    random.shuffle(lst)

prefix = args.prefix
output_dir = args.out
if output_dir[-1] != '/':
    output_dir += '/'

fo = open(args.makefile, "w")

objs = []
cmds = []
fw = None
sz = 0
img_cnt = 1;
cnt = 1

for item in lst:
    if sz + 10240 > (int(args.partition_size)<<20) or fw == None:
        lst_name = output_dir + (prefix % cnt) + '.lst'
        bin_name = output_dir + (prefix % cnt) + '.bin'
        objs.append(bin_name)
        if fw != None:
            fw.close()
        fw = open(lst_name, "w")
        cmd = "%s: %s\n\t%s %s %s %s %s" % (bin_name, lst_name,
                IM2BIN, lst_name, img_root, bin_name, new_size)
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
fw.close()









