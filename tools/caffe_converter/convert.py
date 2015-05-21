import os
import sys
import argparse


# caffe root folder
caffe_root = '/home/winsty/caffe_latest/caffe/'
# cxxnet root folder
cxxnet_root = '/home/winsty/cxxnew/cxxnet/'
sys.path.insert(0, os.path.join(caffe_root, 'python'))
sys.path.insert(0, os.path.join(cxxnet_root, 'wrapper'))

import caffe
import cxxnet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("caffe_prototxt",
                        help="caffe prototxt")
    parser.add_argument(
        "caffe_model", help="caffe model")
    parser.add_argument("cxxnet_conf", help="cxxnet conf")
    parser.add_argument("to_save", help="to save, in format like 0090.model")
    args = parser.parse_args()
    caffe_prototxt = args.caffe_prototxt
    caffe_model = args.caffe_model
    cxxnet_conf = args.cxxnet_conf
    to_save = args.to_save
    print 'converting {0} and {1} with {2} into {3}'.format(caffe_prototxt, caffe_model, cxxnet_conf, to_save)
    caffe.set_mode_cpu()
    net_caffe = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)
    print 'creating cxxnet model'
    with open(cxxnet_conf, 'r') as f_in:
        cfg = f_in.read()
    net_cxxnet = cxxnet.Net(dev='cpu', cfg=cfg)
    net_cxxnet.set_param('dev', 'cpu')
    net_cxxnet.init_model()

    layer_names = net_caffe._layer_names
    first_conv = True
    for layer_idx, layer in enumerate(net_caffe.layers):
        layer_name = layer_names[layer_idx]
        if layer.type == 'Convolution' or layer.type == 'InnerProduct':
            assert(len(layer.blobs) == 2)
            wmat = layer.blobs[0].data
            bias = layer.blobs[1].data
            if first_conv:
                print 'Swapping BGR of caffe into RGB in cxxnet'
                wmat[:, [0, 2], :, :] = wmat[:, [2, 0], :, :]

            assert(wmat.flags['C_CONTIGUOUS'] is True)
            assert(bias.flags['C_CONTIGUOUS'] is True)
            print 'converting layer {0}, wmat shape = {1}, bias shape = {2}'.format(layer_name, wmat.shape, bias.shape)
            wmat = wmat.reshape((wmat.shape[0], -1))
            bias = bias.reshape((bias.shape[0], 1))
            net_cxxnet.set_weight(wmat, layer_name, 'wmat')
            net_cxxnet.set_weight(bias, layer_name, 'bias')
            if first_conv and layer.type == 'Convolution':
                first_conv = False

    net_cxxnet.save_model(to_save)

if __name__ == '__main__':
    main()