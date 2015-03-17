import sys
sys.path.append('../../wrapper/')
import cxxnet
import numpy as np

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

net = cxxnet.train(cfg, data, 1, param, eval_data = deval)

weights = []
for layer in ['fc1', 'fc2']:
    for tag in ['wmat', 'bias']:
        weights.append((layer, tag, net.get_weight(layer, tag)))

data.before_first()
data.next()
# extract 
print 'predict'
pred = net.predict(data)
print 'predict finish'
dbatch = data.get_data()
print dbatch.shape
print 'get data'
pred2 = net.predict(dbatch)

print np.sum(np.abs(pred - pred2))
print np.sum(np.abs(net.extract(data, 'sg1') - net.extract(dbatch, 'sg1')))

# evaluate
deval.before_first()
werr = 0
wcnt = 0
while deval.next():
    label = deval.get_label()
    pred = net.predict(deval)
    werr += np.sum(label[:,0] !=  pred[:])
    wcnt += len(label[:,0])
print 'eval-error=%f' % (float(werr) / wcnt)

# training
data.before_first()
while data.next():
    label = data.get_label()    
    batch = data.get_data()
    net.update(batch, label)

# evaluate
deval.before_first()
werr = 0
wcnt = 0
while deval.next():
    label = deval.get_label()
    pred = net.predict(deval)
    werr += np.sum(label[:,0] !=  pred[:])
    wcnt += len(label[:,0])
print 'eval-error2=%f' % (float(werr) / wcnt)

for layer, tag, wt in weights:
    net.set_weight(wt, layer, tag)

# evaluate
deval.before_first()
werr = 0
wcnt = 0
while deval.next():
    label = deval.get_label()
    pred = net.predict(deval)
    werr += np.sum(label[:,0] !=  pred[:])
    wcnt += len(label[:,0])
print 'eval-error-after-setback=%f' % (float(werr) / wcnt)
