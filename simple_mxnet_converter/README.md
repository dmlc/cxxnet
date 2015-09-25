1. convert conf into symbol python by using ```symbol_converter.py```
2. dump all weight into a new folder by using ```dump_weight.py```
3. add extra script to load weight and build new mxnet symbol (sample at end file of```symbol.py```)

Note:
- It is a toy contains functions I may use, you need to add whatever you need to it
- Only support ```[a->b]``` format in conf
- Remove all inplace conf, eg ```layer[10->10] = softmax:sm``` to ```layer[10->11] = softmax```
- every layer must have NAME

