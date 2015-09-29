from cxxnet import Net
import sys
import re
import numpy as np
LAYER_PATTERN = re.compile(r"layer\[(.*)->(.*)\]\s*=\s*(\w+):(\w*)")
dump_dict = {
        "fullc":["weight", "bias"],
        "conv":["weight", "bias"],
        "batch_norm":["gamma", "beta", "moving_mean", "moving_var"]
}

if len(sys.argv) < 3:
    print("usage: conf model output_folder")

layers = []

fi = open(sys.argv[1])
cfg = ""
for line in fi:
    line = line
    cfg += line
    if LAYER_PATTERN.match(line) != None:
        in_ids_str, out_ids_str, layer, name = LAYER_PATTERN.findall(line)[0]
        layers.append((layer, name))

fi.close()
net = Net(cfg=cfg)
net.init_model()
net.load_model(sys.argv[2])


for layer, name in layers:
    path = sys.argv[3]
    if layer in dump_dict:
        for tag in dump_dict[layer]:
            weight = net.get_weight(name, tag)
            np.save(path + name + '_' + tag, weight)
            if weight == None:
                print name + '_' + tag




