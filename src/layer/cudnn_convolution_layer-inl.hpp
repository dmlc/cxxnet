#ifndef CXXNET_LAYER_CUDNN_CONVOLUTION_LAYER_INL_HPP_
#define CXXNET_LAYER_CUDNN_CONVOLUTION_LAYER_INL_HPP_
#pragma once

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class CuDNNConvolutionLayer : public ConvolutionLayer<xpu> {
 public:
  CuDNNConvolutionLayer(mshadow::Random<xpu> *p_rnd) : ConvolutionLayer<xpu>(p_rnd) {
    use_fast_algo_ = false;
  };
#ifdef __CUDACC__
#if CXXNET_USE_CUDNN == 1
  virtual ~CuDNNConvolutionLayer() {
    CUDA_CHECK(cudnnDestroyTensorDescriptor(in_desc_));
    CUDA_CHECK(cudnnDestroyTensorDescriptor(out_desc_));
    CUDA_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDA_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDA_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
    CUDA_CHECK(cudnnDestroy(handle_));
  };
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    ConvolutionLayer<xpu>::InitNode(nodes_in, nodes_out);
    nodes_in[0]->must_contiguous = true;
    nodes_out[0]->must_contiguous = true;
    this->InitCuDNN();
  }
  virtual void SetParam(const char *name, const char* val) {
    Parent::SetParam(name, val);
    if (!strcmp(name, "algo")) {
      if (!strcmp(val, "fast")) use_fast_algo_ = true;
      else if(!strcmp(val, "balance")) use_fast_algo_ = false;
      else utils::Error("Unkown convolution algo mode");
    }
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    float alpha = 1.0f;
    float beta = 0.0f;
    if (!init_cudnn_) {
      init_cudnn_ = true;
      if (use_fast_algo_) {
        algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
      } else {
        algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      }
      temp_.set_stream(nodes_out[0]->data.stream_);
      CUDA_CHECK(cudnnSetStream(handle_, nodes_out[0]->data.stream_->stream_));
      CUDA_CHECK(cudnnSetFilter4dDescriptor(filter_desc_, dtype_,
                                            Parent::param_.num_channel,
                                            Parent::param_.num_input_channel,
                                            Parent::param_.kernel_height,
                                            Parent::param_.kernel_width));
      CUDA_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_,
                                                 Parent::param_.pad_y,
                                                 Parent::param_.pad_x,
                                                 Parent::param_.stride,
                                                 Parent::param_.stride, 1, 1,
                                                 CUDNN_CROSS_CORRELATION));
      mshadow::Tensor<gpu, 4, float> &in = nodes_in[0]->data;
      mshadow::Tensor<gpu, 4, float> &out = nodes_out[0]->data;
      CUDA_CHECK(cudnnSetTensor4dDescriptor(in_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                            in.shape_[0], in.shape_[1],
                                            in.shape_[2], in.shape_[3]));
      CUDA_CHECK(cudnnSetTensor4dDescriptor(out_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                            out.shape_[0], out.shape_[1],
                                            out.shape_[2], out.shape_[3]));
      CUDA_CHECK(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                            1, Parent::bias_.shape_[0], 1, 1));
      CUDA_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_, in_desc_,
                                                         filter_desc_, conv_desc_,
                                                         out_desc_, algo_,
                                                         &workspace_size_));
      temp_.Resize(mshadow::Shape1(workspace_size_ / sizeof(float) + 1), 0.0f);
    }
    utils::Assert(nodes_in[0]->data.CheckContiguous(), "contiguous in conv");
    utils::Assert(nodes_out[0]->data.CheckContiguous(), "contiguous in conv");
    CUDA_CHECK(cudnnConvolutionForward(handle_, &alpha,
                                       in_desc_, nodes_in[0]->data.dptr_,
                                       filter_desc_, Parent::wmat_.dptr_,
                                       conv_desc_, algo_, temp_.dptr_, workspace_size_, &beta,
                                       out_desc_, nodes_out[0]->data.dptr_));
    if (Parent::param_.no_bias == 0) {
      beta = 1.0f;
      CUDA_CHECK(cudnnAddTensor(handle_, CUDNN_ADD_SAME_C, &alpha,
                                bias_desc_, Parent::bias_.dptr_, &beta,
                                out_desc_, nodes_out[0]->data.dptr_));
    }


  }
  virtual void Backward(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    float alpha = 1.0f;
    float beta = 0.0f;
    if (Parent::param_.no_bias == 0) {
      CUDA_CHECK(cudnnConvolutionBackwardBias(handle_, &alpha,
                                              out_desc_, nodes_out[0]->data.dptr_,
                                              &beta,
                                              bias_desc_, Parent::gbias_.dptr_));
    }
    CUDA_CHECK(cudnnConvolutionBackwardFilter(handle_, &alpha,
                                              in_desc_, nodes_in[0]->data.dptr_,
                                              out_desc_, nodes_out[0]->data.dptr_,
                                              conv_desc_, &beta,
                                              filter_desc_, Parent::gwmat_.dptr_));
    CUDA_CHECK(cudnnConvolutionBackwardData(handle_, &alpha,
                                            filter_desc_, Parent::wmat_.dptr_,
                                            out_desc_, nodes_out[0]->data.dptr_,
                                            conv_desc_, &beta,
                                            in_desc_, nodes_in[0]->data.dptr_));
  }
 private:
  inline void InitCuDNN() {
    init_cudnn_ = false;
    dtype_ = CUDNN_DATA_FLOAT;
    algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    CUDA_CHECK(cudnnCreate(&handle_));
    CUDA_CHECK(cudnnCreateTensorDescriptor(&in_desc_));
    CUDA_CHECK(cudnnCreateTensorDescriptor(&out_desc_));
    CUDA_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDA_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDA_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }
  /*! \brief cuDNN init status */
  bool init_cudnn_;
  /*! \brief cuDNN handle */
  cudnnHandle_t handle_;
  /*! \brief cuDNN data type */
  cudnnDataType_t dtype_;
  /*! \brief cuDNN input tensor descriptor */
  cudnnTensorDescriptor_t in_desc_;
  /*! \brief cuDNN output tensor descriptor */
  cudnnTensorDescriptor_t out_desc_;
  /*! \brief cuDNN bias tensor descriptor */
  cudnnTensorDescriptor_t bias_desc_;
  /*! \brief cuDNN filter descriptor */
  cudnnFilterDescriptor_t filter_desc_;
  /*! \brief cuDNN conv descriptor */
  cudnnConvolutionDescriptor_t conv_desc_;
  /*! \brief cuDNN conv algorithm */
  cudnnConvolutionFwdAlgo_t algo_;
  /*! \brief cuDNN workspace size */
  size_t workspace_size_;
  /*! \brief cuDNN workspace */
  mshadow::TensorContainer<xpu, 1> temp_;
  /*! \brief parent */
  typedef ConvolutionLayer<xpu> Parent;
  /*! \brief whether use fast algorithm */
#endif // CXXNET_USE_CUDNN
#endif // __CUDACC__
  bool use_fast_algo_;
}; // class CuDNNConvolutionLayer
} // namespace layer
} // namespace cxxnet


#endif // CXXNET_LAYER_CUDNN_CONVOLUTION_LAYER_INL_HPP
