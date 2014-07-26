"""
this file contains easy to use utils to write a mshadow matrix
"""
import numpy as np

def write_mshadow(fo, M):
    """
    write a numpy tensor to mshadow format
    """
    mshape = np.zeros(len(M.shape),dtype='uint32')
    # reverse saving the shape
    for i in range(len(M.shape)):
        mshape[i] = M.shape[-(i+1)]
    mshape.tofile(fo)
    M.astype('float32').tofile(fo)
    

def save_data(fname, label, data):
    """
    save label data into fname
    """
    assert len(label.shape)==1
    assert len(data.shape)==4
    fl = open( fname+'.label.mshadow', 'wb')
    write_mshadow( fl, label )
    fl.close()

    fo = open( fname+'.image.mshadow', 'wb')
    write_mshadow( fo, data )
    fo.close()
