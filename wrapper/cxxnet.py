"""
CXXNet python ctypes wrapper
Author: Bing Xu

"""
import ctypes
import os
import sys
import numpy
import numpy.ctypeslib

# set this line correctly
if os.name == 'nt':
    # TODO windows
    CXXNET_PATH = os.path.dirname(__file__) + '/libcxxnetwrapper.dll'
else:
    CXXNET_PATH = os.path.dirname(__file__) + '/libcxxnetwrapper.so'

# load in xgboost library
cxnlib = ctypes.cdll.LoadLibrary(CXXNET_PATH)
cxnlib.CXNIOCreateFromConfig.restype = ctypes.c_void_p
cxnlib.CXNIONext.restype = ctypes.c_int
cxnlib.CXNNetCreate.restype = ctypes.c_void_p
cxnlib.CXNNetPredictBatch.restype = ctypes.POINTER(ctypes.c_float)
cxnlib.CXNNetPredictIter.restype = ctypes.POINTER(ctypes.c_float)
cxnlib.CXNNetEvaluate.restype = ctypes.c_char_p


def ctypes2numpy(cptr, length, dtype=numpy.float32):
    """convert a ctypes pointer array to numpy array """
    #assert isinstance(cptr, ctypes.POINTER(ctypes.c_float))
    res = numpy.zeros(length, dtype=dtype)
    assert ctypes.memmove(res.ctypes.data, cptr, length * res.strides[0])
    return res

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
        ret = cxnlib.CXNIONext(self.handle)
        self.head = False
        self.tail = ret != 0
        return self.tail
    def before_first(self):
        cxnlib.CXNIOBeforeFirst(self.handle)
        self.head = True
        self.tail = False
    def check_valid(self):
        if self.head:
            raise Exception('iterator was at head state, call next to get to valid state')
        if self.tail:
            raise Exception('iterator reaches end')                    
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
                label = label.reshape((label.size(0),1))
            if label.ndim != 2:
                raise Exception('Net.update: label need to be 2 dimension or one dimension ndarray')
            if label.shape[0] != data.shape[0]:
                raise Exception('Net.update: data size mismatch')
            cxnlib.CXNNetUpdateBatch(self.handle,
                                     data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     numpy.array(data.shape).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                                     label.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     numpy.array(label.shape).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)))
        else:
            raise Exception('update do not support type %s' % str(type(data)))

    def evaluate(self, data, name):
        """ update the net using the data
        Parameters
            data: input can be DataIter or numpy.ndarray
            name: str
                name of the input data
        """
        if isinstance(data, DataIter):
            return cxnlib.CXNNetEvaluate(self.handle, data.handle, name)
        else:
            raise Exception('update do not support type %s' % str(type(data)))

    def predict(self, data):
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
                                            numpy.array(data.shape).ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                                            ctypes.byref(olen));
        return ctypes2numpy(ret, olen.value, 'float32')

def train(cfg, data, num_round, param, eval_data = None):
    net = Net(cfg = cfg)
    if isinstance(param, dict):
        param = param.items()
    for k, v in param:
        net.set_param(k, v)
    net.init_model()
    for r in range(num_round):
        net.start_round(r)
        net.update(data)
        if eval_data is not None:
            seval = net.evaluate(eval_data, 'eval')
        sys.stderr.write(seval + '\n')
    return net
