import sys
sys.path.append('../../wrapper/')
import cxxnet

data = cxxnet.DataIter("""
iter = mnist
    path_img = "./data/train-images-idx3-ubyte.gz"
    path_label = "./data/train-labels-idx1-ubyte.gz"
    shuffle = 1
iter = end
input_shape = 1,1,784
batch_size = 100
""")
print 'init data iter'

deval = cxxnet.DataIter("""
iter = mnist
    path_img = "./data/t10k-images-idx3-ubyte.gz"
    path_label = "./data/t10k-labels-idx1-ubyte.gz"
iter = end
input_shape = 1,1,784
batch_size = 100
""")

cfg = """
netconfig=start
layer[+1:fc1] = fullc:fc1
  nhidden = 100
  init_sigma = 0.01
layer[+1:sg1] = sigmoid:se1
layer[sg1->fc2] = fullc:fc2
  nhidden = 10
  init_sigma = 0.01
layer[+0] = softmax
netconfig=end

input_shape = 1,1,784
batch_size = 100

random_type = gaussian
"""

param = {}
param['eta'] = 0.1
param['dev'] = 'cpu'
param['momentum'] = 0.9
param['metric[label]'] = 'error'

net = cxxnet.train(cfg, data, 10, param, eval_data = deval)
