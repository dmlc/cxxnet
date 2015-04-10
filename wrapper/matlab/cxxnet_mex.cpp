
#include <cstdio>
#include <string>
#include <cstring>
#include <assert.h>
#include "mex.h"
#define CXXNET_IN_MATLAB
#include "cxxnet_wrapper.h"


typedef unsigned long long uint64;
union Ptr {
  uint64 data;
  void *ptr;
};


static mxArray* SetHandle(void *handle) {
  union Ptr bridge;
  bridge.data = 0;
  bridge.ptr = handle;
  const mwSize dims[1] = {1};
  mxArray *mx_out = mxCreateNumericArray(1, dims, mxUINT64_CLASS, mxREAL);
  uint64 *up = reinterpret_cast<uint64*>(mxGetData(mx_out));
  *up = bridge.data;
  return mx_out;
}

static void *GetHandle(const mxArray *input) {
  union Ptr bridge;
  uint64 *up = reinterpret_cast<uint64*>(mxGetData(input));
  bridge.data = *up;
  return bridge.ptr;
}

inline void Transpose2D(const cxx_real_t *ptr, cxx_real_t* cxx_ptr,
  cxx_uint oshape[4], cxx_uint ostride) {
  for (cxx_uint i = 0; i < oshape[0]; ++i) {
    for (cxx_uint j = 0; j < oshape[1]; ++j) {
      cxx_ptr[i * ostride + j] = ptr[j * oshape[0] + i];
    }
  }
}

inline void Transpose3D(const cxx_real_t *ptr, cxx_real_t* cxx_ptr,
  cxx_uint oshape[3], cxx_uint ostride) {
  for (cxx_uint i = 0; i < oshape[0]; ++i) {
    for (cxx_uint j = 0; j < oshape[1]; ++j) {
      for (cxx_uint m = 0; m < oshape[2]; ++m) {
        cxx_ptr[i * oshape[1] * ostride + j * ostride + m] 
          = ptr[m * oshape[0] * oshape[1] + j * oshape[0] + i];
      }
    }
  }
}

inline void Transpose4D(const cxx_real_t *ptr, cxx_real_t* cxx_ptr,
  cxx_uint oshape[4], cxx_uint ostride) {
  const cxx_uint cxx_stride = oshape[1] * oshape[2] * ostride;
  const cxx_uint mx_stride = oshape[1] * oshape[2] * oshape[3];
  for (cxx_uint i = 0; i < oshape[0]; ++i) {
    cxx_uint mat_stride_0 = oshape[0];
    cxx_real_t *inst_cxx_ptr = cxx_ptr + i * cxx_stride;
    for (cxx_uint j = 0; j < oshape[1]; ++j) {
      cxx_uint mat_stride_1 = mat_stride_0 * oshape[1];
      cxx_real_t *mat_cxx_ptr = inst_cxx_ptr + j * oshape[2] * ostride;
      for (cxx_uint m = 0; m < oshape[2]; ++m) {
        cxx_uint mat_stride_2 = mat_stride_1 * oshape[2];
        for (cxx_uint n = 0; n < oshape[3]; ++n) {
          mat_cxx_ptr[m * ostride + n] 
            = ptr[n * mat_stride_2 + m * mat_stride_1 + j * mat_stride_0 + i];
        }
      }
    }
  }
}
        
inline mxArray* Ctype2Mx4DT(const cxx_real_t *ptr, cxx_uint oshape[4], cxx_uint ostride) {
  // COL MAJOR PROBLEM
  const mwSize dims[4] = {oshape[0], oshape[1], oshape[2], oshape[3]};
  const cxx_uint cxx_stride = oshape[1] * oshape[2] * ostride;
  const cxx_uint mx_stride = oshape[1] * oshape[2] * oshape[3];
  mxArray *mx_out = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
  cxx_real_t *mx_ptr = reinterpret_cast<cxx_real_t*>(mxGetData(mx_out));
  for (cxx_uint i = 0; i < oshape[0]; ++i) {
    cxx_uint mat_stride_0 = oshape[0];
    cxx_real_t *inst_cxx_ptr = const_cast<cxx_real_t*>(ptr) + i * cxx_stride;
    for (cxx_uint j = 0; j < oshape[1]; ++j) {
      cxx_uint mat_stride_1 = mat_stride_0 * oshape[1];
      cxx_real_t *mat_cxx_ptr = inst_cxx_ptr + j * oshape[2] * ostride;
      for (cxx_uint m = 0; m < oshape[2]; ++m) {
        cxx_uint mat_stride_2 = mat_stride_1 * oshape[2];
        for (cxx_uint n = 0; n < oshape[3]; ++n) {
          mx_ptr[n * mat_stride_2 + m * mat_stride_1 + j * mat_stride_0 + i]
                  = mat_cxx_ptr[m * ostride + n];
        }
      }
    }
  }
  return mx_out;
}

inline mxArray* Ctype2Mx3DT(const cxx_real_t *ptr, cxx_uint oshape[3], cxx_uint ostride) {
  // COL MAJOR PROBLEM
  const mwSize dims[4] = {oshape[0], oshape[1], oshape[2]};
  mxArray *mx_out = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
  cxx_real_t *mx_ptr = reinterpret_cast<cxx_real_t*>(mxGetData(mx_out));
  for (cxx_uint i = 0; i < oshape[0]; ++i) {
    for (cxx_uint j = 0; j < oshape[1]; ++j) {
      for (cxx_uint m = 0; m < oshape[2]; ++m) {
        mx_ptr[m * oshape[0] * oshape[1] + j * oshape[0] + i]
          = ptr[i * oshape[1] * ostride + j * ostride + m];
      }
    }
  }
  return mx_out;
}

inline mxArray* Ctype2Mx2DT(const cxx_real_t *ptr, cxx_uint oshape[2], cxx_uint ostride) {
  const mwSize dims[2] = {oshape[0], oshape[1]};
  mxArray *mx_out = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
  cxx_real_t *mx_ptr = reinterpret_cast<cxx_real_t*>(mxGetPr(mx_out));
  for (cxx_uint i = 0; i < oshape[0]; ++i) {
    for (cxx_uint j = 0; j < oshape[1]; ++j) {
      mx_ptr[j * oshape[0] + i] = ptr[i * ostride + j];
    }
  }
  return mx_out;
}


inline mxArray* Ctype2Mx1DT(const cxx_real_t *ptr, cxx_uint len) {
  const mwSize dims[1] = {len};
  mxArray *mx_out = mxCreateNumericArray(1, dims, mxSINGLE_CLASS, mxREAL);
  cxx_real_t *mx_ptr = reinterpret_cast<cxx_real_t*>(mxGetData(mx_out));
  memcpy(mx_ptr, ptr, len * sizeof(cxx_real_t));
  return mx_out;
}



static void MEXCXNIOCreateFromConfig(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  char *conf = mxArrayToString(prhs[1]);
  void *handle = CXNIOCreateFromConfig(conf);
  plhs[0] = SetHandle(handle);
  mxFree(conf);
}

static void MEXCXNIONext(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  const mwSize dims[1] = {1};
  mxArray *mx_out = mxCreateNumericArray(1, dims, mxINT32_CLASS, mxREAL);
  int *mx_ptr = reinterpret_cast<int*>(mxGetData(mx_out));
  int res = CXNIONext(handle);
  memcpy(mx_ptr, &res, sizeof(int));
  plhs[0] = mx_out;
}

static void MEXCXNIOBeforeFirst(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  CXNIOBeforeFirst(handle);
}

static void MEXCXNIOGetData(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  cxx_uint oshape[4];
  cxx_uint ostride = 0;
  const cxx_real_t *res_ptr = CXNIOGetData(handle, oshape, &ostride);
  plhs[0] = Ctype2Mx4DT(res_ptr, oshape, ostride);
}

static void MEXCXNIOGetLabel(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  cxx_uint oshape[2];
  cxx_uint ostride = 0;
  const cxx_real_t *res_ptr = CXNIOGetLabel(handle, oshape, &ostride);
  plhs[0] = Ctype2Mx2DT(res_ptr, oshape, ostride);
}

static void MEXCXNIOFree(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  CXNIOFree(handle);
}

static void MEXCXNNetCreate(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  char *dev = mxArrayToString(prhs[1]);
  char *conf = mxArrayToString(prhs[2]);
  void *handle = CXNNetCreate(dev, conf);
  plhs[0] = SetHandle(handle);
  mxFree(dev);
  mxFree(conf);
}

static void MEXCXNNetFree(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  CXNNetFree(handle);
}

static void MEXCXNNetSetParam(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  char *key = mxArrayToString(prhs[2]);
  char *val = mxArrayToString(prhs[3]);
  CXNNetSetParam(handle, key, val);
  mxFree(key);
  mxFree(val);
}

static void MEXCXNNetInitModel(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  CXNNetInitModel(handle);
}

static void MEXCXNNetSaveModel(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  char *fname = mxArrayToString(prhs[2]);
  CXNNetSaveModel(handle, fname);
  mxFree(fname);
}

static void MEXCXNNetLoadModel(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  char *fname = mxArrayToString(prhs[2]);
  CXNNetLoadModel(handle, fname);
  mxFree(fname);
}

static void MEXCXNNetStartRound(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  int *ptr = reinterpret_cast<int*>(mxGetData(prhs[2]));
  CXNNetStartRound(handle, *ptr);
}

static void MEXCXNNetSetWeight(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_weight = prhs[2];
  char *layer_name = mxArrayToString(prhs[3]);
  char *wtag = mxArrayToString(prhs[4]);
  cxx_uint size = 1;
  cxx_real_t *ptr = reinterpret_cast<cxx_real_t*>(mxGetData(p_weight));
  cxx_uint dims = mxGetNumberOfDimensions(p_weight);
  const mwSize *mx_shape = mxGetDimensions(p_weight);
  cxx_uint shape[4];
  for (cxx_uint i = 0; i < dims; ++i) {
    size *= mx_shape[i];
    shape[i] = mx_shape[i];
  }
  cxx_real_t *cxx_ptr = NULL;
  switch (dims){
    case 2:
      cxx_ptr = new cxx_real_t[shape[0] * shape[1]];
      Transpose2D(ptr, cxx_ptr, shape, shape[1]);
      break;
    case 3:
      cxx_ptr = new cxx_real_t[shape[0] * shape[1] * shape[2]];
      Transpose3D(ptr, cxx_ptr, shape, shape[2]);
      break;
    case 4:
      cxx_ptr = new cxx_real_t[shape[0] * shape[1] * shape[2] * shape[3]];
      Transpose4D(ptr, cxx_ptr, shape, shape[3]);
      break;
  }
  CXNNetSetWeight(handle, cxx_ptr, size, layer_name, wtag);
  mxFree(layer_name);
  mxFree(wtag);
  delete[] cxx_ptr;
}

static void MEXCXNNetGetWeight(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  char *layer_name = mxArrayToString(prhs[2]);
  char *wtag = mxArrayToString(prhs[3]);
  cxx_uint wshape[4] = {0};
  cxx_uint odim = 0;
  const cxx_real_t *res_ptr = CXNNetGetWeight(handle, layer_name, wtag, wshape, &odim);
  if (odim == 0) res_ptr = NULL;
  if (wshape[3] != 0) {
    plhs[0] = Ctype2Mx4DT(res_ptr, wshape, wshape[3]);
  } else if (wshape[2] != 0) {
    cxx_uint shape[3] = {wshape[0], wshape[1], wshape[2]};
    plhs[0] = Ctype2Mx3DT(res_ptr, shape, shape[2]);
  } else if (wshape[1] != 0) {
    cxx_uint shape[2] = {wshape[0], wshape[1]};
    plhs[0] = Ctype2Mx2DT(res_ptr, shape, shape[1]);
  } else {
    plhs[0] = Ctype2Mx1DT(res_ptr, wshape[0]);
  }
  mxFree(layer_name);
  mxFree(wtag);
}

static void MEXCXNNetUpdateIter(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  CXNNetUpdateIter(handle, data_handle);
}

static void MEXCXNNetUpdateBatch(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_data = prhs[2];
  const mxArray *p_label = prhs[3];
  cxx_uint dshape[4];
  cxx_uint lshape[2];
  assert(mxGetNumberOfDimensions(p_data) == 4);
  assert(mxGetNumberOfDimensions(p_label) == 2);
  const mwSize *d_size = mxGetDimensions(p_data);
  const mwSize *l_size = mxGetDimensions(p_label);
  for (int i = 0; i < 4; ++i) dshape[i] = d_size[i];
  for (int i = 0; i < 2; ++i) lshape[i] = l_size[i];
  cxx_real_t *ptr_data = reinterpret_cast<cxx_real_t*>(mxGetData(p_data));
  cxx_real_t *ptr_label = reinterpret_cast<cxx_real_t*>(mxGetData(p_label));
  cxx_real_t *cxx_ptr_data = new cxx_real_t[dshape[0] * dshape[1] * dshape[2] * dshape[3]];
  cxx_real_t *cxx_ptr_label = new cxx_real_t[lshape[0] * lshape[1]];
  Transpose4D(ptr_data, cxx_ptr_data, dshape, dshape[3]);
  Transpose2D(ptr_label, cxx_ptr_label, lshape, lshape[1]);
  CXNNetUpdateBatch(handle, cxx_ptr_data, dshape, cxx_ptr_label, lshape);
  delete[] cxx_ptr_data;
  delete[] cxx_ptr_label;
}

static void MEXCXNNetPredictBatch(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_data = prhs[2];
  cxx_uint dshape[4];
  assert(mxGetNumberOfDimensions(p_data) == 4);
  const mwSize *d_size = mxGetDimensions(p_data);
  for (int i = 0; i < 4; ++i) dshape[i] = d_size[i];
  cxx_uint out_size = 0;
  cxx_real_t *ptr_data = reinterpret_cast<cxx_real_t*>(mxGetData(p_data));
  cxx_real_t *cxx_ptr_data = new cxx_real_t[dshape[0] * dshape[1] * dshape[2] * dshape[3]];
  Transpose4D(ptr_data, cxx_ptr_data, dshape, dshape[3]);
  const cxx_real_t *ptr_res = CXNNetPredictBatch(handle, cxx_ptr_data, dshape, &out_size);
  plhs[0] = Ctype2Mx1DT(ptr_res, out_size);
  delete[] cxx_ptr_data;
}

static void MEXCXNNetPredictIter(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  cxx_uint out_size = 0;
  const cxx_real_t *ptr_res = CXNNetPredictIter(handle, data_handle, &out_size);
  plhs[0] = Ctype2Mx1DT(ptr_res, out_size);
}

static void MEXCXNNetExtractBatch(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  const mxArray *p_data = prhs[2];
  char *node_name = mxArrayToString(prhs[3]);
  assert(mxGetNumberOfDimensions(p_data) == 4);
  cxx_uint dshape[4];
  const mwSize *d_size = mxGetDimensions(p_data);
  for (int i = 0; i < 4; ++i) dshape[i] = d_size[i];
  cxx_uint oshape[4];
  cxx_real_t *ptr_data = reinterpret_cast<cxx_real_t*>(mxGetData(p_data));
  cxx_real_t *cxx_ptr_data = new cxx_real_t[dshape[0] * dshape[1] * dshape[2] * dshape[3]];
  Transpose4D(ptr_data, cxx_ptr_data, dshape, dshape[3]);
  const cxx_real_t *ptr_res = CXNNetExtractBatch(handle, cxx_ptr_data, dshape, node_name, oshape);
  plhs[0] = Ctype2Mx4DT(ptr_res, oshape, oshape[3]);
  mxFree(node_name);
  delete[] cxx_ptr_data;
}

static void MEXCXNNetExtractIter(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  char *node_name = mxArrayToString(prhs[3]);
  cxx_uint oshape[4];
  const cxx_real_t *ptr_res = CXNNetExtractIter(handle, data_handle, node_name, oshape);
  plhs[0] = Ctype2Mx4DT(ptr_res, oshape, oshape[3]);
  mxFree(node_name);
}

static void MEXCXNNetEvaluate(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  void *handle = GetHandle(prhs[1]);
  void *data_handle = GetHandle(prhs[2]);
  char *data_name = mxArrayToString(prhs[3]);
  const char *ret = CXNNetEvaluate(handle, data_handle, data_name);
  printf("%s\n", ret);
  plhs[0] = mxCreateString(ret);
  mxFree(data_name);
}


// MEX Function
//

struct handle_registry {
  std::string cmd;
  void (*func)(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs);
};


static handle_registry handles[] = {
  {"MEXCXNIOCreateFromConfig", MEXCXNIOCreateFromConfig},
  {"MEXCXNIONext", MEXCXNIONext},
  {"MEXCXNIOBeforeFirst", MEXCXNIOBeforeFirst},
  {"MEXCXNIOGetData", MEXCXNIOGetData},
  {"MEXCXNIOGetLabel", MEXCXNIOGetLabel},
  {"MEXCXNIOFree", MEXCXNIOFree},
  {"MEXCXNNetCreate", MEXCXNNetCreate},
  {"MEXCXNNetFree", MEXCXNNetFree},
  {"MEXCXNNetSetParam", MEXCXNNetSetParam},
  {"MEXCXNNetInitModel", MEXCXNNetInitModel},
  {"MEXCXNNetSaveModel", MEXCXNNetSaveModel},
  {"MEXCXNNetLoadModel", MEXCXNNetLoadModel},
  {"MEXCXNNetStartRound", MEXCXNNetStartRound},
  {"MEXCXNNetSetWeight", MEXCXNNetSetWeight},
  {"MEXCXNNetGetWeight", MEXCXNNetGetWeight},
  {"MEXCXNNetUpdateIter", MEXCXNNetUpdateIter},
  {"MEXCXNNetUpdateBatch", MEXCXNNetUpdateBatch},
  {"MEXCXNNetPredictBatch", MEXCXNNetPredictBatch},
  {"MEXCXNNetPredictIter", MEXCXNNetPredictIter},
  {"MEXCXNNetExtractBatch", MEXCXNNetExtractBatch},
  {"MEXCXNNetExtractIter", MEXCXNNetExtractIter},
  {"MEXCXNNetEvaluate", MEXCXNNetEvaluate},
  {"NULL", NULL},
};

void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
  mexLock();  // Avoid clearing the mex file.
  if (nrhs == 0) {
    mexErrMsgTxt("No API command given");
    return;
  }
  char *cmd = mxArrayToString(prhs[0]);
  bool dispatched = false;
  for (int i = 0; handles[i].func != NULL; i++) {
    if (handles[i].cmd.compare(cmd) == 0) {
      handles[i].func(nlhs, plhs, nrhs, prhs);
      dispatched = true;
      break;
    }
  }
  if (!dispatched) {
    std::string err = "Unknown command '";
    err += cmd;
    err += "'";
    mexErrMsgTxt(err.c_str());
  }
  mxFree(cmd);
}
