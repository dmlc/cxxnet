import os
from collections import defaultdict
from copy import deepcopy
graph = defaultdict(set)

class Param:
    layer = None
    kernel_height = None
    kernel_width = None
    pad = 0
    pad_x = None
    pad_y = None
    nhidden = None
    nchannel = None
    threshold = None
    stride = 1
    kernel_size = 1
    def __init__(self, *param, **kwargs):
        for dic in param:
            for key in dic:
                setattr(self, key, dic[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.NotJoin = []
        if self.pad_x == None:
            self.pad_x = self.pad
            self.NotJoin.append("pad_x")
        if self.pad_y == None:
            self.pad_y = self.pad
            self.NotJoin.append("pad_y")
        if self.kernel_width == None:
            self.kernel_width = self.kernel_size
            self.NotJoin.append("kernel_width")
        if self.kernel_height == None:
            self.kernel_height = self.kernel_size
            self.NotJoin.append("kernel_height")
    def join(self):
        cmd = ""
        items = self.__dict__.items()
        for item in items:
            if item[0] in self.NotJoin:
                continue
            if item[0] == "NotJoin":
                continue
            if item[0] == "layer":
                continue
            if item[0] == "pad_x" and item[1] == 0:
                continue
            if item[0] == "pad_y" and item[1] == 0:
                continue
            if item[0] == "kernel_width" and item[1] == 1:
                continue
            if item[0] == "kernel_height" and item[1] == 1:
                continue
            cmd += "  %s=%s\n" % (item[0], str(item[1]))
        return cmd



class Node:
    cnt = [0]
    def __init__(self):
        self.idx = self.cnt[0]
        self.cnt[0] += 1
        self.ch = -1
        self.x = -1
        self.y = -1
        self.has_shape = False
    def SetShape(self, ch, y, x):
        self.ch = ch
        self.x = x
        self.y = y
        self.has_shape = True


class Layer:
    cnt = defaultdict(int)
    def __init__(self, param):
        self.attached = None
        self.in_nodes = []
        self.out_nodes = []
        self.param = param
        self.cnt[param.layer] += 1
        self.idx = self.cnt[param.layer]
        self.name = "%s_%s" % (self.param.layer, str(self.idx))
    def Print(self):
        cmd_in = ",".join([str(n.idx) for n in self.in_nodes])
        cmd_out = ",".join([str(n.idx) for n in self.out_nodes])
        if len(cmd_out) == 0:
            cmd_out = cmd_in
        cmd = "layer[%s->%s] = %s:%s\n%s" % \
                (cmd_in, cmd_out, self.param.layer, self.name, self.param.join())
        if self.attached != None:
            cmd += self.attached.Print()
        return cmd
    def VZ(self):
        info = ["shape=box", "style=filled", "fixedsize=true", "width=1.1", "height=0.6798"]
        if self.param.layer == "conv":
            info.append("color=royalblue1")
            info.append('label="convolution\n%dx%d/%d, %d"' % (self.param.kernel_height, \
                    self.param.kernel_width, self.param.stride, self.param.nchannel))
        elif self.param.layer == "fullc":
            info.append("color=royalblue1")
            info.append('label="fullc\n%d"' % self.param.nhidden)
        elif self.param.layer == "batch_norm":
            info.append("color=orchid1")
            info.append('label="batch_norm"')
        elif "concat" in self.param.layer:
            info.append("color=seagreen1")
            info.append('label=%s' % self.param.layer)
        elif self.param.layer == "split":
            info.append("color=seagreen1")
            info.append('label=%s' % self.param.layer)
        elif self.param.layer == "flatten":
            info.append("color=seagreen1")
            info.append('label=%s' % self.param.layer)
        elif "pooling" in self.param.layer:
            info.append("color=firebrick2")
            info.append('label="%s\n%dx%d/%d"' % (self.param.layer, self.param.kernel_height, \
                    self.param.kernel_width, self.param.stride))
        elif "elu" in self.param.layer:
            info.append("color=salmon")
            info.append('label=%s' % self.param.layer)
        else:
            info.append("color=olivedrab1")
            info.append('label=%s' % self.param.layer)
        return "%s [%s];\n" % (self.name, ",".join(info))





def AddConnection(conn1, conn2):
    if conn2.param.layer == "dropout":
        conn2.in_nodes = conn1.out_nodes
        conn2.out_nodes = conn1.out_nodes
        conn1.attached = conn2
        return
    nd = Node()
    conn1.out_nodes.append(nd)
    conn2.in_nodes.append(nd)
    global graph
    graph[conn1].add(conn2)

def ConnectToData(conn, data):
    conn.in_nodes.append(data)
    global graph
    graph[data].add(conn)

def CheckConnection(conn):
    if conn.param.layer == "fullc":
        assert(conn.param.nhidden != None)
        assert(len(conn.out_nodes) == 1)
        assert(len(conn.in_nodes) == 1)
        if conn.in_nodes[0].has_shape:
            assert(conn.in_nodes[0].ch == 1 and conn.in_nodes[0].y == 1)
            conn.out_nodes[0].SetShape(1, 1, conn.param.nhidden)
    elif conn.param.layer == "flatten":
        assert(len(conn.out_nodes) == 1)
        assert(len(conn.in_nodes) == 1)
        if conn.in_nodes[0].has_shape:
            conn.out_nodes[0].SetShape(1, 1, conn.in_nodes[0].ch * conn.in_nodes[0].x * conn.in_nodes[0].y)
    elif conn.param.layer == "max_pooling" or conn.param.layer == "avg_pooling":
        assert((conn.param.kernel_height != None and conn.param.kernel_width != None) or \
                conn.param.kernel_size != None)
        assert(len(conn.out_nodes) == 1)
        assert(len(conn.in_nodes) == 1)
        if conn.in_nodes[0].has_shape:
            ch = conn.in_nodes[0].ch
            x = conn.in_nodes[0].x
            y = conn.in_nodes[0].y
            x = min(x + 2 * conn.param.pad_x - conn.param.kernel_width + conn.param.stride - 1, \
                    x + 2 * conn.param.pad_x - 1) / conn.param.stride + 1
            y = min(y + 2 * conn.param.pad_y - conn.param.kernel_height + conn.param.stride - 1, \
                    y + 2 * conn.param.pad_y - 1) / conn.param.stride + 1
            assert(x > 0)
            assert(y > 0)
            conn.out_nodes[0].SetShape(ch, y, x)
    elif conn.param.layer == "conv":
        assert((conn.param.kernel_height != None and conn.param.kernel_width != None) or \
                conn.param.kernel_size != None)
        assert(len(conn.out_nodes) == 1)
        assert(len(conn.in_nodes) == 1)
        if conn.in_nodes[0].has_shape:
            assert(conn.param.nchannel != None)
            ch = conn.param.nchannel
            x = conn.in_nodes[0].x
            y = conn.in_nodes[0].y
            x = (x + 2 * conn.param.pad_x - conn.param.kernel_width) / conn.param.stride + 1
            y = (y + 2 * conn.param.pad_y - conn.param.kernel_height) / conn.param.stride + 1
            assert(x > 0)
            assert(y > 0)
            conn.out_nodes[0].SetShape(ch, y, x)
    elif conn.param.layer == "split":
        assert(len(conn.in_nodes) == 1)
        assert(len(conn.out_nodes) >= 1)
        if conn.in_nodes[0].has_shape:
            for nd in conn.out_nodes:
                nd.SetShape(conn.in_nodes[0].ch, conn.in_nodes[0].y, conn.in_nodes[0].x)

    elif conn.param.layer == "concat":
        assert(len(conn.in_nodes) > 0)
        if conn.in_nodes[0].has_shape:
            x = 0
            y = conn.in_nodes[0].y
            ch = conn.in_nodes[0].ch
            for i in xrange(len(conn.in_nodes)):
                assert(conn.in_nodes[i].ch == conn.in_nodes[0].ch)
                assert(conn.in_nodes[i].y == conn.in_nodes[0].y)
                x += conn.in_nodes[i].x
            conn.out_nodes[0].SetShape(ch, y, x)
    elif conn.param.layer == "ch_concat":
        assert(len(conn.in_nodes) > 0)
        if conn.in_nodes[0].has_shape:
            x = conn.in_nodes[0].x
            y = conn.in_nodes[0].y
            ch = 0
            for i in xrange(len(conn.in_nodes)):
                assert(conn.in_nodes[i].x == conn.in_nodes[0].x)
                assert(conn.in_nodes[i].y == conn.in_nodes[0].y)
                ch += conn.in_nodes[i].ch
            conn.out_nodes[0].SetShape(ch, y, x)
    elif conn.param.layer == "flatten":
        if conn.in_nodes[0].has_shape:
            x = conn.in_nodes[0].x
            y = conn.in_nodes[0].y
            ch = conn.in_nodes[0].ch
            conn.out_nodes[0].SetShape(1, 1, x * y * ch)
    else:
        if conn.in_nodes[0].has_shape:
            x = conn.in_nodes[0].x
            y = conn.in_nodes[0].y
            ch = conn.in_nodes[0].ch
            if len(conn.out_nodes) > 0:
                conn.out_nodes[0].SetShape(ch, y, x)

def ConvFactory(nchannel, kernel_size=1, pad=0, stride = 1, bn = True, act = "relu"):
    tmp = []
    tmp.append(Layer(Param({"layer":"conv", "pad":pad, "nchannel":nchannel, "stride":stride, "kernel_size":kernel_size})))
    if bn:
        tmp.append(Layer(Param({"layer":"batch_norm"})))
    tmp.append(Layer(Param({"layer":act})))
    for i in xrange(1, len(tmp)):
        AddConnection(tmp[i-1], tmp[i])
    return tmp

def DFS(layer, table, seq):
    out_layers = graph[layer]
    for l in out_layers:
        DFS(l, table, seq)
    if layer not in table:
        table.add(layer)
        seq.append(layer)

def Generate(layer):
    table = set([])
    seq = []
    DFS(layer, table, seq)
    seq = seq[::-1]
    idx = 1
    table = set([])
    for c in seq:
        for nd in c.out_nodes:
            if nd not in table:
                table.add(nd)
                nd.idx = idx
                idx += 1
    conf = ""
    for c in seq:
        print "Init %s" % c.name
        CheckConnection(c)
        try:
            print "output size: %d-%d-%d" % (c.out_nodes[0].ch, c.out_nodes[0].y, c.out_nodes[0].x)
        except:
            pass
        conf += c.Print()
    return conf

def Graphviz(layer, show_size=True):
    table = set([])
    seq = []
    DFS(layer, table, seq)
    seq = seq[::-1]
    info = "data [shape=box, fixedsize=true, width=1.1, height=0.6798];\n"
    dot = "digraph G {\n"
    if show_size == True:
        dot += '%s -> data [dir="back", label="%sx%sx%s"];\n' % (seq[0].name,
                str(layer.in_nodes[0].ch), str(layer.in_nodes[0].y), str(layer.in_nodes[0].x))
    else:
        dot += '%s -> data [dir="back"];\n' % seq[0].name
    for c in seq:
        info += c.VZ()
        conns = graph[c]
        for nxt in conns:
            sz = ""
            if show_size == True:
                try:
                    sz = "%dx%dx%d" % (c.out_nodes[0].ch, c.out_nodes[0].y, c.out_nodes[0].x)
                except:
                    pass
            cmd = '%s -> %s [dir="back", label="%s"];\n' % (nxt.name, c.name, sz)
            dot += cmd
    dot += info
    dot += "}\n"
    return dot

#############################################################################################################
#
#   Modify your network from here
#   Advise for write factory: return a list, first element is input layer, last element is output layer
#
#############################################################################################################
def FactoryInception(ch_1x1, ch_3x3r, ch_3x3, ch_3x3dr, ch_3x3d, ch_proj, act="rrelu", stride = 1):
    param = {}
    split = Layer(Param({"layer":"split"}))
    concat = Layer(Param({"layer":"ch_concat"}))

    #1x1
    if stride != 2:
        # Manual assemble layers
        param["layer"] = "conv"
        param["stride"] = 1
        param["pad"] = 0
        param["kernel_size"] = 1
        param["nchannel"] = ch_1x1

        conv1x1 = Layer(Param(param))
        bn1x1 = Layer(Param({"layer":"batch_norm"}))
        act1x1 = Layer(Param({"layer":act}))
        AddConnection(split, conv1x1)
        AddConnection(conv1x1, bn1x1)
        AddConnection(bn1x1, act1x1)
        AddConnection(act1x1, concat)
    #3x3reduce + 3x3
    # Use exist factory
    conv3x3r = ConvFactory(stride= 1, pad = 0, kernel_size = 1, nchannel = ch_3x3r, bn = True, act = act)
    conv3x3 = ConvFactory(stride= stride, pad = 1, kernel_size = 3, nchannel = ch_3x3, bn = True, act = act)
    AddConnection(split, conv3x3r[0])
    AddConnection(conv3x3r[-1], conv3x3[0])
    AddConnection(conv3x3[-1], concat)

    #double 3x3reduce + double 3x3
    conv3x3dr = ConvFactory(stride= 1, pad = 0, kernel_size = 1, nchannel = ch_3x3dr, bn = True, act = act)
    conv3x3d = ConvFactory(stride = stride, pad = 1, kernel_size = 3, nchannel = ch_3x3d, bn = True, act = act)
    AddConnection(split, conv3x3dr[0])
    AddConnection(conv3x3dr[-1], conv3x3d[0])
    AddConnection(conv3x3d[-1], concat)

    # pool + project
    if stride == 1:
        param["layer"] = "avg_pooling"
        param["stride"] = 1
        param["pad"] = 1
        param["kernel_size"] = 3
        del(param["nchannel"])
        pool = Layer(Param(param))

        param["layer"] = "conv"
        param["stride"] = 1
        param["pad"] = 0
        param["kernel_size"] = 1
        param["nchannel"] = ch_proj
        proj2 = Layer(Param(param))
        bn2 = Layer(Param({"layer":"batch_norm"}))
        act2 = Layer(Param({"layer":"relu"}))
        AddConnection(split, pool)
        AddConnection(pool, proj2)
        AddConnection(proj2, bn2)
        AddConnection(bn2, act2)
        AddConnection(act2, concat)
    else:
        param["layer"] = "max_pooling"
        param["stride"] = stride
        param["pad"] = 0
        param["kernel_size"] = 3
        try:
            del(param["nchannel"])
        except:
            pass
        pool = Layer(Param(param))
        AddConnection(split, pool)
        AddConnection(pool, concat)

    return [split, concat]




Factory = FactoryInception

data = Node()
data.SetShape(3,224,224)
conv1 = ConvFactory(kernel_size = 7, stride = 2, pad = 3, nchannel = 64, bn = True, act = "rrelu")
pool1 = Layer(Param({"layer":"max_pooling", "kernel_size":3, "stride":2}))
conv2a = ConvFactory(kernel_size = 1, stride = 1, pad = 0, nchannel = 64, bn = True, act = "rrelu")
conv2b = ConvFactory(kernel_size = 3, stride = 1, pad = 1, nchannel = 192, bn = True, act = "rrelu")
pool2 = Layer(Param({"layer":"max_pooling", "kernel_size":3, "stride":2}))
in3a = Factory(64, 64, 64, 64, 96, 32)
in3b = Factory(64, 64, 96, 64, 96, 64)
in3c = Factory(0, 128, 160, 64, 96, 0, stride=2)
in4a = Factory(224, 64, 96, 96, 128, 128)
in4b = Factory(192, 96, 128, 96, 128, 128)
in4c = Factory(160, 128, 160, 128, 160, 128)
in4d = Factory(96, 128, 192, 160, 192, 128)
in4e = Factory(0, 128, 192, 192, 256, 0, stride = 2)
in5a = Factory(352, 192, 320, 160, 224, 128)
in5b = Factory(352, 192, 320, 192, 224, 128)
avg = Layer(Param({"layer":"avg_pooling", "kernel_size":7, "stride":1}))
flatten = Layer(Param({"layer":"flatten"}))
fc = Layer(Param({"layer":"fullc", "nhidden":1000}))
loss = Layer(Param({"layer":"softmax"}))


ConnectToData(conv1[0], data)
AddConnection(conv1[-1], pool1)
AddConnection(pool1, conv2a[0])
AddConnection(conv2a[-1], conv2b[0])
AddConnection(conv2b[-1], pool2)
AddConnection(pool2, in3a[0])
AddConnection(in3a[-1], in3b[0])
AddConnection(in3b[-1], in3c[0])
AddConnection(in3c[-1], in4a[0])
AddConnection(in4a[-1], in4b[0])
AddConnection(in4b[-1], in4c[0])
AddConnection(in4c[-1], in4d[0])
AddConnection(in4d[-1], in4e[0])
AddConnection(in4e[-1], in5a[0])
AddConnection(in5a[-1], in5b[0])
AddConnection(in5b[-1], avg)
AddConnection(avg, flatten)
AddConnection(flatten, fc)
AddConnection(fc, loss)

conf = ""
conf +=  Generate(conv1[0])
fo = open("inception.conf", "w")
fo.write(conf)
fo.close()
dot = Graphviz(conv1[0])
fw = open("inception.gv", "w")
fw.write(dot)
fw.close()
os.system("dot -Tpng inception.gv -o inception.png")
