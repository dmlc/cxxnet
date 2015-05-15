"""
CXXNet python ctypes wrapper
Author: Tianqi Chen, Bing Xu

"""
import ctypes
import os
import sys
import numpy
import numpy.ctypeslib

# set this line correctly
if os.name == 'nt':
    # TODO windows
    CXXNET_PATH = os.path.join(os.path.dirname(__file__), 'libcxxnetwrapper.dll')
else:
    CXXNET_PATH = os.path.join(os.path.dirname(__file__), 'libcxxnetwrapper.so')

# load in xgboost library
cxnlib = ctypes.cdll.LoadLibrary(CXXNET_PATH)
cxnlib.CXNIOCreateFromConfig.restype = ctypes.c_void_p
cxnlib.CXNIONext.restype = ctypes.c_int
cxnlib.CXNIOGetData.restype = ctypes.POINTER(ctypes.c_float)
cxnlib.CXNIOGetLabel.restype = ctypes.POINTER(ctypes.c_float)
cxnlib.CXNNetCreate.restype = ctypes.c_void_p
cxnlib.CXNNetPredictBatch.restype = ctypes.POINTER(ctypes.c_float)
cxnlib.CXNNetPredictIter.restype = ctypes.POINTER(ctypes.c_float)
cxnlib.CXNNetExtractBatch.restype = ctypes.POINTER(ctypes.c_float)
cxnlib.CXNNetExtractIter.restype = ctypes.POINTER(ctypes.c_float)
cxnlib.CXNNetGetWeight.restype = ctypes.POINTER(ctypes.c_float)
cxnlib.CXNNetEvaluate.restype = ctypes.c_char_p


def ctypes2numpy(cptr, length, dtype=numpy.float32):
    """convert a ctypes pointer array to numpy array """
    #assert isinstance(cptr, ctypes.POINTER(ctypes.c_float))
    res = numpy.zeros(length, dtype=dtype)
    if not ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0]):
        raise AssertionError('ctypes.memmove failed')
    return res

def ctypes2numpyT(cptr, shape, dtype=numpy.float32, stride = None):
    """convert a ctypes pointer array to numpy array """
    size = 1
    for x in shape:
        size *= x
    if stride is None:
        res = numpy.zeros(size, dtype=dtype)
        if not ctypes.memmove(res.ctypes.data, cptr, size * res.strides[0]):
            raise AssertionError('ctypes.memmove failed')
    else:
        dsize = size / shape[-1] * stride
        res = numpy.zeros(dsize, dtype=dtype)
        if not ctypes.memmove(res.ctypes.data, cptr, dsize * res.strides[0]):
            raise AssertionError('ctypes.memmove failed')
        res = res.reshape((dsize / shape[-1], shape[-1]))
        res = res[:, 0 :shape[-1]]
    return res.reshape(shape)

def shape2ctypes(data):
    shape = (ctypes.c_uint * data.ndim)()
    for i in range(data.ndim):
        shape[i] = data.shape[i]
    return shape


class DataIter:
    """data iterator of cxxnet"""
    def __init__(self, cfg):
        self.handle = cxnlib.CXNIOCreateFromConfig(ctypes.c_char_p(cfg.encode('utf-8')))
        self.head = True
        self.tail = False
    def __del__(self):
        """destructor"""
        cxnlib.CXNIOFree(self.handle)
    def next(self):
        """next batch in iter"""
        ret = cxnlib.CXNIONext(self.handle)
        self.head = False
        self.tail = ret == 0
        return ret != 0
    def before_first(self):
        """reset iterator"""
        cxnlib.CXNIOBeforeFirst(self.handle)
        self.head = True
        self.tail = False
    def check_valid(self):
        """check iterator state"""
        if self.head:
            raise Exception('iterator was at head state, call next to get to valid state')
        if self.tail:
            raise Exception('iterator reaches end')
    def get_data(self):
        """get current batch data"""
        oshape = (ctypes.c_uint * 4)()
        ostride = ctypes.c_uint()
        ret = cxnlib.CXNIOGetData(self.handle,
                                  oshape, ctypes.byref(ostride))
        return ctypes2numpyT(ret, [x for x in oshape], 'float32', ostride.value)
    def get_label(self):
        """get current batch label"""
        oshape = (ctypes.c_uint * 2)()
        ostride = ctypes.c_uint()
        ret = cxnlib.CXNIOGetLabel(self.handle,
                                   oshape, ctypes.byref(ostride))
        return ctypes2numpyT(ret, [x for x in oshape], 'float32', ostride.value)

class Net:
    """neural net object"""
    def __init__(self, dev = 'cpu', cfg = ''):
        self.handle = cxnlib.CXNNetCreate(ctypes.c_char_p(dev.encode('utf-8')),
                                          ctypes.c_char_p(cfg.encode('utf-8')))

    def __del__(self):
        """destructor"""
        cxnlib.CXNNetFree(self.handle)

    def set_param(self, name, value):
        """set paramter to the trainer"""
        name = str(name)
        value = str(value)
        cxnlib.CXNNetSetParam(self.handle,
                              ctypes.c_char_p(name.encode('utf-8')),
                              ctypes.c_char_p(value.encode('utf-8')))

    def init_model(self):
        """ initialize the network structure
        """
        cxnlib.CXNNetInitModel(self.handle)

    def load_model(self, fname):
        """ load model from file
        Parameters
            fname: str
                name of model
        """
        cxnlib.CXNNetLoadModel(self.handle, fname)

    def save_model(self, fname):
        """ save model to file
        Parameters
            fname: str
                name of model
        """
        cxnlib.CXNNetSaveModel(self.handle, fname)

    def start_round(self, round_counter):
        """ notify the net the training phase of round counter begins
        Parameters
            round_counter: int
                current round counter
        """
        cxnlib.CXNNetStartRound(self.handle, round_counter)

    def update(self, data, label = None):
        """ update the net using the data
        Parameters
            data: input can be DataIter or numpy.ndarray
            label: the label of the data batch
        """
        if isinstance(data, DataIter):
            data.check_valid()
            cxnlib.CXNNetUpdateIter(self.handle, data.handle)
        elif isinstance(data, numpy.ndarray):
            if data.ndim != 4:
                raise Exception('Net.update: need 4 dimensional tensor (batch, channel, height, width)')
            if label is None:
                raise Exception('Net.update: need label to use update')
            if not isinstance(label, numpy.ndarray):
                raise Exception('Net.update: label need to be ndarray')
            if label.ndim == 1:
                label = label.reshape(label.shape[0], 1)
            if label.ndim != 2:
                raise Exception('Net.update: label need to be 2 dimension or one dimension ndarray')
            if label.shape[0] != data.shape[0]:
                raise Exception('Net.update: data size mismatch')
            if data.dtype != numpy.float32:
                raise Exception('Net.update: data must be of type numpy.float32')
            if label.dtype != numpy.float32:
                raise Exception('Net.update: label must be of type numpy.float32')
            cxnlib.CXNNetUpdateBatch(self.handle,
                                     data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     shape2ctypes(data),
                                     label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     shape2ctypes(label))
        else:
            raise Exception('update do not support type %s' % str(type(data)))

    def evaluate(self, data, name):
        """ evaluate the model using data iterator
        Parameters
            data: input can be DataIter
            name: str
                name of the input data
        Return:
            Evaluation string
        """
        if isinstance(data, DataIter):
            return cxnlib.CXNNetEvaluate(self.handle, data.handle, name)
        else:
            raise Exception('update do not support type %s' % str(type(data)))

    def predict(self, data):
        """ make prediction from data
        Parameters
            data: iter or numpy ndarray
        Return
            prediction in numpy array
        """
        olen = ctypes.c_uint()
        if isinstance(data, DataIter):
            data.check_valid()
            ret = cxnlib.CXNNetPredictIter(self.handle,
                                           data.handle,
                                           ctypes.byref(olen));
        elif isinstance(data, numpy.ndarray):
            if data.ndim != 4:
                raise Exception('need 4 dimensional tensor to use predict')

            ret = cxnlib.CXNNetPredictBatch(self.handle,
                                            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                            shape2ctypes(data),
                                            ctypes.byref(olen));
        return ctypes2numpy(ret, olen.value, 'float32')

    def extract(self, data, name):
        """Extract feature from data
        Parameters
            data: iter or numpy ndarray
            name: node name to be extracted
        Return
            feature in numpy array
        """
        oshape = (ctypes.c_uint * 4)()
        if isinstance(data, DataIter):
            data.check_valid()
            ret = cxnlib.CXNNetExtractIter(self.handle,
                                           data.handle,
                                           ctypes.c_char_p(name.encode('utf-8')),
                                           oshape);
        elif isinstance(data, numpy.ndarray):
            if data.ndim != 4:
                raise Exception('need 4 dimensional tensor to use extract')
            ret = cxnlib.CXNNetExtractBatch(self.handle,
                                            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                            shape2ctypes(data),
                                            ctypes.c_char_p(name.encode('utf-8')),
                                            oshape)
        return ctypes2numpyT(ret, [x for x in oshape], 'float32')

    def set_weight(self, weight, layer_name, tag):
        """Set weight for special layer
        Parameters
            weight: new weight array
            layer_name: layer to be set
            tag: bias or wmat
        """
        if tag != 'bias' and tag != 'wmat':
            raise Exception('tag must be bias or wmat')
        cxnlib.CXNNetSetWeight(self.handle,
                               weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               weight.size,
                               ctypes.c_char_p(layer_name.encode('utf-8')),
                               ctypes.c_char_p(tag.encode('utf-8')))

    def get_weight(self, layer_name, tag):
        """Get weight array from layer
           Parameter
                layer_name: name of layer
                tag: bias or wmat
           return
                weight array

        """

        if tag != 'bias' and tag != 'wmat':
            raise Exception('tag must be bias or wmat')
        oshape = (ctypes.c_uint * 4)()
        odim = ctypes.c_uint()
        ret = cxnlib.CXNNetGetWeight(self.handle,
                                     ctypes.c_char_p(layer_name.encode('utf-8')),
                                     ctypes.c_char_p(tag.encode('utf-8')),
                                     oshape, ctypes.byref(odim))
        if odim.value == 0 or ret is None:
            return None
        return ctypes2numpyT(ret, [oshape[i] for i in range(odim.value)], 'float32')

def train(cfg, data, label, num_round, param, eval_data = None):
    net = Net(cfg = cfg)
    if isinstance(param, dict):
        param = param.items()
    for k, v in param:
        net.set_param(k, v)
    net.init_model()
    if isinstance(data, DataIter):
        for r in range(num_round):
            net.start_round(r)
            data.before_first()
            scounter = 0
            while data.next():
                net.update(data)
                scounter += 1
                if scounter % 100  == 0:
                    print '[%d] %d batch passed' % (r, scounter)
            if eval_data is not None:
                seval = net.evaluate(eval_data, 'eval')
            sys.stderr.write(seval + '\n')
        return net
    else:
        for r in range(num_round):
            print "Training in round %d" % r
            net.start_round(r)
            net.update(data=data, label=label)
        return net


