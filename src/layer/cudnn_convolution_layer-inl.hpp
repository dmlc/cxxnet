#ifndef CXXNET_LAYER_CUDNN_CONVOLUTION_LAYER_INL_HPP_
#define CXXNET_LAYER_CUDNN_CONVOLUTION_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class CuDNNConvolutionLayer : public ILayer<xpu> {
 public:
  CuDNNConvolutionLayer(mshadow::Random<xpu> *p_rnd)
      : prnd_(p_rnd), wmat_(false), bias_(false), gwmat_(false), gbias_(false) {}
  virtual ~CuDNNConvolutionLayer(void) {
    CUDA_CHECK(cudnnDestroy(handle_));
    CUDA_CHECK(cudnnDestroyTensorDescriptor(in_desc_));
    CUDA_CHECK(cudnnDestroyTensorDescriptor(out_desc_));
    CUDA_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDA_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDA_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam(name, val);
  }
  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit("wmat", wmat_, gwmat_);
    if (param_.no_bias == 0) {
      pvisitor->Visit("bias", bias_, gbias_);
    }
  }
  virtual void InitModel(void) {
    wmat_.Resize(mshadow::Shape4(param_.num_channel, param_.num_input_channel, \
                                 param_.kernel_height, param_.kernel_width));
    bias_.Resize(mshadow::Shape4(1, param_.num_channel, 1, 1), false);
    param_.RandInitWeight(this->prnd_, wmat_, wmat_.size(0), wmat_.size(1));
    bias_ = param_.init_bias;
    gwmat_.Resize(wmat_.shape_);
    gbias_.Resize(bias_.shape_);
    gwmat_ = 0.0f; gbias_ = 0.0f;
  }
  virtual void SaveModel(utils::IStream &fo) const {
    fo.Write(&param_, sizeof(LayerParam));
    wmat_.SaveBinary(fo);
    bias_.SaveBinary(fo);
  }
  virtual void LoadModel(utils::IStream &fi) {
    utils::Check(fi.Read(&param_, sizeof(LayerParam)) != 0,
                  "ConvolutionLayer: LoadModel invalid model file");
    wmat_.LoadBinary(fi);
    bias_.LoadBinary(fi);
    // setup gradient
    gwmat_.Resize(wmat_.shape_, false);
    gbias_.Resize(bias_.shape_, false);
    gwmat_ = 0.0f; gbias_ = 0.0f;
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    // stream of wmat and bias may be reset, but it is ok
    wmat_.set_stream(stream);
    bias_.set_stream(stream);
    gwmat_.set_stream(stream);
    gbias_.set_stream(stream);
    temp_.set_stream(stream);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "ConvolutionLayer Layer only support 1-1 connection");
    const index_t ksize_y = static_cast<index_t>(param_.kernel_height);
    const index_t ksize_x = static_cast<index_t>(param_.kernel_width);
    const index_t kstride = static_cast<index_t>(param_.stride);
    mshadow::Shape<4> ishape = nodes_in[0]->data.shape_;
    utils::Check(param_.num_group == 1, "CuDNN Conv layer only support 1 group now.");
    utils::Check(ishape[1] % param_.num_group == 0,  "input channels must divide group size");
    utils::Check(param_.num_channel % param_.num_group == 0, "output channels must divide group size");
    utils::Check(param_.num_channel > 0, "must set nchannel correctly");
    utils::Check(param_.kernel_height > 0 && param_.kernel_width > 0, "must set kernel_size correctly");
    utils::Check(ksize_x <= ishape[3] && ksize_y <= ishape[2], "kernel size exceed input");
    mshadow::Shape<4> oshape = mshadow::
        Shape4(ishape[0], param_.num_channel,
                (ishape[2] + 2 * param_.pad_y - ksize_y) / kstride + 1,
                (ishape[3] + 2 * param_.pad_x - ksize_x) / kstride + 1);
    nodes_out[0]->data.shape_ = oshape;
    oheight_ = oshape[2];
    owidth_ = oshape[3];
    if (param_.num_input_channel == 0) {
      param_.num_input_channel = static_cast<int>(ishape[1]);
    } else {
      utils::Check(param_.num_input_channel == static_cast<int>(ishape[1]),
                   "ConvolutionLayer: number of input channels is not consistent");
    }
    init_cudnn_ = false;
    alpha_ = 1.0f;
    beta_ = 0.0f;
    dtype_ = CUDNN_DATA_FLOAT;
    algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    CUDA_CHECK(cudnnCreate(&handle_));
    CUDA_CHECK(cudnnCreateTensorDescriptor(&in_desc_));
    CUDA_CHECK(cudnnCreateTensorDescriptor(&out_desc_));
    CUDA_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDA_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDA_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    if (!init_cudnn_) {
      init_cudnn_ = true;
      CUDA_CHECK(cudnnSetStream(handle_, nodes_out[0]->data.stream_->stream_));
      CUDA_CHECK(cudnnSetFilter4dDescriptor(filter_desc_, dtype_,
                                 param_.num_channel,
                                 param_.num_input_channel,
                                 param_.kernel_height,
                                 param_.kernel_width));

      CUDA_CHECK(cudnnSetConvolution2dDescriptor(conv_desc_, param_.pad_y, param_.pad_x,
                                     param_.stride, param_.stride, 1, 1, CUDNN_CROSS_CORRELATION));
      mshadow::Tensor<gpu, 4, float> &sgt = nodes_in[0]->data;
      mshadow::Tensor<gpu, 4, float> &dgt = nodes_out[0]->data;
      CUDA_CHECK(cudnnSetTensor4dDescriptorEx(in_desc_, dtype_,
                                   sgt.shape_[0], sgt.shape_[1], sgt.shape_[2], sgt.shape_[3],
                                   sgt.stride_ * sgt.shape_[1] * sgt.shape_[2],
                                   sgt.stride_ * sgt.shape_[2],
                                   sgt.stride_,
                                   1));
      CUDA_CHECK(cudnnSetTensor4dDescriptorEx(out_desc_, dtype_,
                                   dgt.shape_[0], dgt.shape_[1], dgt.shape_[2], dgt.shape_[3],
                                   dgt.stride_ * dgt.shape_[1] * dgt.shape_[2],
                                   dgt.stride_ * dgt.shape_[2],
                                   dgt.stride_,
                                   1));

      CUDA_CHECK(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                  bias_.shape_[0], bias_.shape_[1],
                                  bias_.shape_[2], bias_.shape_[3]));


      CUDA_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_, in_desc_,
                                              filter_desc_, conv_desc_,
                                              out_desc_, algo_,
                                              &workspace_size_));
      temp_.Resize(mshadow::Shape1(workspace_size_ / sizeof(float) + 1), 0.0f);
    }
    beta_ = 0.0f;
    CUDA_CHECK(cudnnConvolutionForward(handle_, &alpha_,
                            in_desc_, nodes_in[0]->data.dptr_,
                            filter_desc_, wmat_.dptr_,
                            conv_desc_, algo_, temp_.dptr_, workspace_size_, &beta_,
                            out_desc_, nodes_out[0]->data.dptr_));
    if (param_.no_bias == 0) {
      beta_ = 1.0f;
      CUDA_CHECK(cudnnAddTensor(handle_, CUDNN_ADD_SAME_C, &alpha_,
                    bias_desc_, bias_.dptr_, &beta_,
                    out_desc_, nodes_out[0]->data.dptr_));
    }
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    if (param_.no_bias == 0) {
      CUDA_CHECK(cudnnConvolutionBackwardBias(handle_, &alpha_,
                                  out_desc_, nodes_out[0]->data.dptr_,
                                  &beta_,
                                  bias_desc_, gbias_.dptr_));
    }

    CUDA_CHECK(cudnnConvolutionBackwardFilter(handle_, &alpha_,
                             in_desc_, nodes_in[0]->data.dptr_,
                             out_desc_, nodes_out[0]->data.dptr_,
                             conv_desc_, &beta_,
                             filter_desc_, gwmat_.dptr_));
    CUDA_CHECK(cudnnConvolutionBackwardData(handle_, &alpha_,
                                 filter_desc_, wmat_.dptr_,
                                 out_desc_, nodes_out[0]->data.dptr_,
                                 conv_desc_, &beta_,
                                 in_desc_, nodes_in[0]->data.dptr_));

  }

 private:

  /*! \brief random number generator */
  mshadow::Random<xpu> *prnd_;
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
  /*! \brief weight matrix */
  mshadow::TensorContainer<xpu,4> wmat_;
  /*! \brief bias */
  mshadow::TensorContainer<xpu,4> bias_;
  /*! \brief accumulates the gradient of weight matrix */
  mshadow::TensorContainer<xpu,4> gwmat_;
  /*! \brief accumulates the gradient of bias */
  mshadow::TensorContainer<xpu,4> gbias_;
  /*! \brief cuDNN init status */
  bool init_cudnn_;
  /*! \brief alpha param for cuDNN */
  float alpha_;
  /*! \brief beta param for cuDNN */
  float beta_;
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
  /*! \brief output height */
  mshadow::index_t oheight_;
  /*! \brief output width */
  mshadow::index_t owidth_;

};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_CONVOLUTION_LAYER_INL_HPP_
