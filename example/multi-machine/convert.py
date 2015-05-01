import re
import sys
import math
import argparse


parser = argparse.ArgumentParser(description='random type converter for distributed cxxnet')
parser.add_argument('--input_conf', required=True, help="path to input conf file")
parser.add_argument('--output_conf', required=True, help="path to output conf")
parser.add_argument('--type', default="xavier", help="use [xavier/kaiming] for convert. default: xavier")
parser.add_argument('--a', default="0", help="extra bias for init")

args = parser.parse_args()

class Param:
    def __init__(self):
        self.clear()
    def clear(self):
        self.kernel_size = 1
        self.nchannel = 1
        self.nhidden = 1
        self.type = None

task = args.type


def gen_str(p, a):
    res = ""
    if p.type == 'conv' or p.type == 'fullc':
        if task == "kaiming":
            res += "  random_type=gaussian\n"
            res += "  init_sigma="
        else:
            res += "  random_type=uniform\n"
            res += "  init_uniform="
        res += str(math.sqrt(2.0 / (1 + a * a) / p.kernel_size / p.kernel_size / p.nchannel / p.nhidden))
        res += '\n'
    return res



param = Param()

START = re.compile(r"\s*netconfig\s*=\s*start\s*")
LAYER = re.compile(r"\s*layer\[.*\]\s*=\s*(\w+):*\w*\s*")
END = re.compile(r"\s*netconfig\s*=\s*end\s*")

KERNEL_SIZE = re.compile(r"\s*kernel_size\s*=\s*(\d+)\s*")
NCHANNEL = re.compile(r"\s*nchannel\s*=\s*(\d+)\s*")
NHIDDEN = re.compile(r"\s*nhidden\s*=\s*(\d+)\s*")

extra = int(args.a)
state = 0
fi = file(args.input_conf)
fo = open(args.output_conf, "w")

cfg = [line for line in fi]
loc = 0

while loc < len(cfg):
    line = cfg[loc]
    if state == 0: # outside network conf
        pass
    elif state == 1: # inside network conf
        if len(LAYER.findall(line)) > 0:
            param.clear()
            param.type = LAYER.findall(line)[0]
    elif state == 2: # inside layer
        if len(KERNEL_SIZE.findall(line)) > 0:
            param.kernel_size = int(KERNEL_SIZE.findall(line)[0])
        if len(NCHANNEL.findall(line)) > 0:
            param.nchannel = int(NCHANNEL.findall(line)[0])
        if len(NHIDDEN.findall(line)) > 0:
            param.nhidden = int(NHIDDEN.findall(line)[0])
    if state == 0:
        if START.match(line) != None:
            state = 1
        fo.write(line)
        loc += 1
    elif state == 1:
        if len(LAYER.findall(line)) > 0:
            state = 2
        fo.write(line)
        loc += 1
    elif state == 2:
        if LAYER.match(line) != None or END.match(line) != None:
            res = gen_str(param, extra)
            fo.write(res)
        else:
            loc += 1
        if END.match(line) != None:
            state = 0
            loc += 1
        if LAYER.match(line) != None:
            state = 1
        fo.write(line)





