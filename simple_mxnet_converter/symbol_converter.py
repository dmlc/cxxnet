import re
from collections import defaultdict
import sys
import os

print("Note: Please remove all inplace setting in cxxnet configure file, only only support [a->b] format")

print("I only implmented some I needed, you have to implment what you need")

if len(sys.argv) < 2:
    print ("usage: in.conf out.py")

LAYER_PATTERN = re.compile(r"layer\[(.*)->(.*)\]\s*=\s*(\w+):(\w*)")
# output: source ids, target ids, layer type, name
PARAM_PATTERN = re.compile(r"\s*(\w+)\s*=\s*(\w+)\s*")
# output: key, value
ID_PATTERN = re.compile(r"([^,]+)")
# output id
CONF_START_PATTERN = re.compile(r"\s*netconfig\s*=\s*start\s*")
CONF_END_PATTERN = re.compile(r"\s*netconfig\s*=\s*end\s*")



id2name = {"0":"data"}
name2def = {"data":"mx.symbol.Variable"}
symbol_param = defaultdict(list)
edge = defaultdict(list)
seq = ["data"]
last_name = "data"


def ParamFactory(key, value):
    if key == "kernel_size":
        return "kernel=(%s, %s)" % (value, value)
    elif key == "nchannel":
        return "num_filter=%s" % value
    elif key == "pad":
        return "pad=(%s, %s)" % (value, value)
    elif key == "stride":
        return "stride=(%s, %s)" % (value, value)
    elif key == "nhidden":
        return "num_hidden=%s" % value
    else:
        return "%s=%s" % (key, value)

def SymbolFactory(layer, name):
    if layer == "conv":
        return "mx.symbol.Convolution"
    if layer == "max_pooling":
        symbol_param[name].append("pool_type='max'")
        return "mx.symbol.Pooling"
    if layer == "avg_pooling":
        symbol_param[name].append("pool_type='avg'")
        return "mx.symbol.Pooling"
    if layer == "relu":
        symbol_param[name].append("act_type='relu'")
        return "mx.symbol.Activation"
    if layer == "rrelu":
        symbol_param[name].append("act_type='rrelu'")
        return "mx.symbol.LeakyReLU"
    if layer == "batch_norm":
        return "mx.symbol.BatchNorm"
    if layer == "ch_concat":
        return "mx.symbol.Concat"
    if layer == "flatten":
        return "mx.symbol.Flatten"
    if layer == "fullc":
        return "mx.symbol.FullyConnected"
    if layer == "softmax":
        return "mx.symbol.Softmax"

def InOutFactory(in_ids_str, out_ids_str, name):
    in_ids = ID_PATTERN.findall(in_ids_str)
    out_ids = ID_PATTERN.findall(out_ids_str)
    # split
    if len(in_ids) == 1 and len(out_ids) > 1:
        for out_id in out_ids:
            id2name[out_id] = id2name[in_ids[0]]
    else:
        # lazy
        id2name[out_ids[0]] = name
        seq.append(name)
        for out_id in out_ids:
            for in_id in in_ids:
                try:
                    edge[id2name[out_id]].append(id2name[in_id])
                except:
                    print id2name
                    raise ValueError("")

def SymbolBuilder(name):
    sym = name2def[name]
    # data
    inputs = edge[name]
    if len(inputs) == 0:
        data = None
        symbol_param[name].append("name='%s'" % name)
    elif len(inputs) == 1:
        data = "%s" % inputs[0]
    else:
        # concat
        data = None
        symbol_param[name].append("*[%s]" % (",".join(inputs)))
    if data != None:
        symbol_param[name].append("data=%s" % data)
    params = ",".join(symbol_param[name])
    cmd = "%s = %s(%s)" % (name, sym, params)
    return cmd

in_conf_flag = False
fi = file(sys.argv[1])
for line in fi:
    if CONF_START_PATTERN.match(line) != None:
        in_conf_flag = True
        continue
    if CONF_END_PATTERN.match(line) != None:
        in_conf_flag = False
    if not in_conf_flag:
        continue
    if LAYER_PATTERN.match(line) != None:
        in_ids_str, out_ids_str, layer, name = LAYER_PATTERN.findall(line)[0]
        last_name = name
        name2def[name] = SymbolFactory(layer, name)
        InOutFactory(in_ids_str, out_ids_str, name)
        symbol_param[name].append("name='%s'" % name)
    elif PARAM_PATTERN.match(line) != None:
        key, value = PARAM_PATTERN.findall(line)[0]
        symbol_param[last_name].append(ParamFactory(key, value))


fo = open(sys.argv[2], "w")
fo.write("import mxnet as mx\n")
for name in seq:
    fo.write(SymbolBuilder(name) + '\n')
fo.close()

