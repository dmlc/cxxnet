#ifndef CXXNET_LAYER_CUDNN_CONVOLUTION_LAYER_INL_HPP_
#define CXXNET_LAYER_CUDNN_CONVOLUTION_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class CuDNNConvolutionLayer : public ConvolutionLayer<xpu> {
 public:
  CuDNNConvolutionLayer(mshadow::Random<xpu> *p_rnd)
      : ConvolutionLayer<xpu>(p_rnd) {
  }
};
#if defined(__CUDACC__) && !defined(CXXNET_COMPILE_CPU_)
template<>
class CuDNNConvolutionLayer<gpu> : public ConvolutionLayer<gpu> {
 public:
  CuDNNConvolutionLayer(mshadow::Random<gpu> *p_rnd) : ConvolutionLayer<gpu>(p_rnd) {
    use_fast_algo_ = false;
  };
#if CXXNET_USE_CUDNN == 1
  virtual ~CuDNNConvolutionLayer() {
    utils::Check(cudnnDestroyTensorDescriptor(in_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnDestroyTensorDescriptor(out_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnDestroyTensorDescriptor(bias_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnDestroyFilterDescriptor(filter_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnDestroyConvolutionDescriptor(conv_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnDestroy(handle_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
  };
  virtual void InitConnection(const std::vector<Node<gpu>*> &nodes_in,
                              const std::vector<Node<gpu>*> &nodes_out,
                              ConnectState<gpu> *p_cstate) {
    ConvolutionLayer<gpu>::InitNode(nodes_in, nodes_out);
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
    if (!strcmp(name, "ngroup")) {
        if (atoi(val) != 1) {
            utils::Error("Currently implementation does not support group when using CuDNN.\
                    Please disable CuDNN and try again.");
        }
    }
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<gpu>*> &nodes_in,
                       const std::vector<Node<gpu>*> &nodes_out,
                       ConnectState<gpu> *p_cstate) {
    float alpha = 1.0f;
    float beta = 0.0f;
    if (!init_cudnn_) {
      init_cudnn_ = true;
      temp_.set_stream(nodes_out[0]->data.stream_);
      utils::Check(cudnnSetStream(handle_, nodes_out[0]->data.stream_->stream_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnSetFilter4dDescriptor(filter_desc_, dtype_,
                                            Parent::param_.num_channel,
                                            Parent::param_.num_input_channel,
                                            Parent::param_.kernel_height,
                                            Parent::param_.kernel_width) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnSetConvolution2dDescriptor(conv_desc_,
                                                 Parent::param_.pad_y,
                                                 Parent::param_.pad_x,
                                                 Parent::param_.stride,
                                                 Parent::param_.stride, 1, 1,
                                                 CUDNN_CROSS_CORRELATION) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      mshadow::Tensor<gpu, 4, float> &in = nodes_in[0]->data;
      mshadow::Tensor<gpu, 4, float> &out = nodes_out[0]->data;
      utils::Check(cudnnSetTensor4dDescriptor(in_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                            in.shape_[0], in.shape_[1],
                                            in.shape_[2], in.shape_[3]) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnSetTensor4dDescriptor(out_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                            out.shape_[0], out.shape_[1],
                                            out.shape_[2], out.shape_[3]) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                            1, Parent::bias_.shape_[0], 1, 1) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      // cudnn v3
      utils::Check(cudnnGetConvolutionForwardAlgorithm(handle_,
                                                       in_desc_,
                                                       filter_desc_,
                                                       conv_desc_,
                                                       out_desc_,
                                                       CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                       512<<20,
                                                       &algo_) == CUDNN_STATUS_SUCCESS, "cudnn fail");

      utils::Check(cudnnGetConvolutionBackwardFilterAlgorithm(handle_,
                                                              in_desc_,
                                                              out_desc_,
                                                              conv_desc_,
                                                              filter_desc_,
                                                              CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                              512<<20,
                                                              &back_algo_w_) == CUDNN_STATUS_SUCCESS, "cudnn fail");

      utils::Check(cudnnGetConvolutionBackwardDataAlgorithm(handle_,
                                                              filter_desc_,
                                                              out_desc_,
                                                              conv_desc_,
                                                              in_desc_,
                                                              CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                              512<<20,
                                                              &back_algo_) == CUDNN_STATUS_SUCCESS, "cudnn fail");
      size_t back_size = 0;
      size_t back_size_w = 0;
      utils::Check(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_,
                                                                filter_desc_,
                                                                out_desc_,
                                                                conv_desc_,
                                                                in_desc_,
                                                                back_algo_,
                                                                &back_size) == CUDNN_STATUS_SUCCESS, "cudnn fail");
      utils::Check(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_,
                                                                  in_desc_,
                                                                  out_desc_,
                                                                  conv_desc_,
                                                                  filter_desc_,
                                                                  back_algo_w_,
                                                                  &back_size_w) == CUDNN_STATUS_SUCCESS, "cudnn fail");
      back_size = std::max(back_size, back_size_w);
      utils::Check(cudnnGetConvolutionForwardWorkspaceSize(handle_, in_desc_,
                                                         filter_desc_, conv_desc_,
                                                         out_desc_, algo_,
                                                         &workspace_size_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      workspace_size_ = std::max(back_size, workspace_size_);
      temp_.Resize(mshadow::Shape1(workspace_size_ / sizeof(float) + 1), 0.0f);
    }
    CHECK(nodes_in[0]->data.CheckContiguous());
    CHECK(nodes_out[0]->data.CheckContiguous());
    utils::Check(cudnnConvolutionForward(handle_, &alpha,
                                       in_desc_, nodes_in[0]->data.dptr_,
                                       filter_desc_, Parent::wmat_.dptr_,
                                       conv_desc_, algo_, temp_.dptr_, workspace_size_, &beta,
                                       out_desc_, nodes_out[0]->data.dptr_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    if (Parent::param_.no_bias == 0) {
      beta = 1.0f;
      utils::Check(cudnnAddTensor(handle_, CUDNN_ADD_SAME_C, &alpha,
                                bias_desc_, Parent::bias_.dptr_, &beta,
                                out_desc_, nodes_out[0]->data.dptr_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    }


  }
  virtual void Backward(bool prop_grad,
                        const std::vector<Node<gpu>*> &nodes_in,
                        const std::vector<Node<gpu>*> &nodes_out,
                        ConnectState<gpu> *p_cstate) {
    float alpha = 1.0f;
    float beta = 0.0f;
    if (Parent::param_.no_bias == 0) {
      utils::Check(cudnnConvolutionBackwardBias(handle_, &alpha,
                                              out_desc_, nodes_out[0]->data.dptr_,
                                              &beta,
                                              bias_desc_, Parent::gbias_.dptr_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    }
    utils::Check(cudnnConvolutionBackwardFilter_v3(handle_, &alpha,
                                              in_desc_, nodes_in[0]->data.dptr_,
                                              out_desc_, nodes_out[0]->data.dptr_,
                                              conv_desc_, back_algo_w_,
                                              temp_.dptr_, workspace_size_,
                                              &beta,
                                              filter_desc_, Parent::gwmat_.dptr_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnConvolutionBackwardData_v3(handle_, &alpha,
                                            filter_desc_, Parent::wmat_.dptr_,
                                            out_desc_, nodes_out[0]->data.dptr_,
                                            conv_desc_, back_algo_,
                                            temp_.dptr_, workspace_size_,
                                            &beta,
                                            in_desc_, nodes_in[0]->data.dptr_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
  }
 private:
  inline void InitCuDNN() {
    init_cudnn_ = false;
    dtype_ = CUDNN_DATA_FLOAT;
    algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    utils::Check(cudnnCreate(&handle_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnCreateTensorDescriptor(&in_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnCreateTensorDescriptor(&out_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnCreateTensorDescriptor(&bias_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnCreateFilterDescriptor(&filter_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    utils::Check(cudnnCreateConvolutionDescriptor(&conv_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
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
  /*! \brief cuDNN back algo for data */
  cudnnConvolutionBwdDataAlgo_t back_algo_;
  /*! \brief cuDNN back algo for filter */
  cudnnConvolutionBwdFilterAlgo_t back_algo_w_;
  /*! \brief cuDNN workspace size */
  size_t workspace_size_;
  /*! \brief cuDNN workspace */
  mshadow::TensorContainer<gpu, 1> temp_;
  /*! \brief parent */
  typedef ConvolutionLayer<gpu> Parent;
  /*! \brief whether use fast algorithm */
#endif // CXXNET_USE_CUDNN
  bool use_fast_algo_;
}; // class CuDNNConvolutionLayer
#endif // __CUDACC__ && ! CXXNET_COMPILE_CPU_
} // namespace layer
} // namespace cxxnet

#endif // CXXNET_LAYER_CUDNN_CONVOLUTION_LAYER_INL_HPP_
