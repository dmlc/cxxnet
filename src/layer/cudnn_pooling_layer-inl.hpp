#ifndef CXXNET_LAYER_CUDNN_POOLING_LAYER_INL_HPP_
#define CXXNET_LAYER_CUDNN_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"

namespace cxxnet {
namespace layer {

template<typename Reducer, int mode, typename xpu>
class CuDNNPoolingLayer : public PoolingLayer<Reducer, mode, xpu> {
  private:
    typedef PoolingLayer<Reducer, mode, xpu> Parent;
  public:
    CuDNNPoolingLayer(){}
#ifdef __CUDACC__
#if CXXNET_USE_CUDNN == 1
  public:
    virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                                const std::vector<Node<xpu>*> &nodes_out,
                                ConnectState<xpu> *p_cstate) {
      Parent::InitNode(nodes_in, nodes_out, p_cstate);
      this->InitCuDNN();
      nodes_in[0]->must_contiguous = true;
      nodes_out[0]->must_contiguous = true;
    }
    virtual ~CuDNNPoolingLayer(void) {
      CUDA_CHECK(cudnnDestroyTensorDescriptor(in_desc_));
      CUDA_CHECK(cudnnDestroyTensorDescriptor(out_desc_));
      CUDA_CHECK(cudnnDestroyPoolingDescriptor(pooling_desc_));
      CUDA_CHECK(cudnnDestroy(handle_));
    }

    virtual void Forward(bool is_train,
                         const std::vector<Node<xpu>*> &nodes_in,
                         const std::vector<Node<xpu>*> &nodes_out,
                         ConnectState<xpu> *p_cstate) {
      mshadow::Tensor<xpu,4> &tmp = p_cstate->states[0];
      if (!init_cudnn_) {
        init_cudnn_ = true;
        CUDA_CHECK(cudnnSetStream(handle_, nodes_out[0]->data.stream_->stream_));
        mshadow::Tensor<gpu, 4, float> &in = nodes_in[0]->data;
        mshadow::Tensor<gpu, 4, float> &out = nodes_out[0]->data;
        CUDA_CHECK(cudnnSetTensor4dDescriptor(in_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                              in.shape_[0], in.shape_[1],
                                              in.shape_[2], in.shape_[3]));
        CUDA_CHECK(cudnnSetTensor4dDescriptor(out_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                              out.shape_[0], out.shape_[1],
                                              out.shape_[2], out.shape_[3]));
      }
      float alpha = 1.0f;
      float beta = 0.0f;
      utils::Assert(nodes_in[0]->data.CheckContiguous(), "contiguous in conv");
      utils::Assert(nodes_out[0]->data.CheckContiguous(), "contiguous in conv");
      utils::Assert(tmp.CheckContiguous(), "contiguous in conv");
      CUDA_CHECK(cudnnPoolingForward(handle_, pooling_desc_, &alpha,
                                     in_desc_, nodes_in[0]->data.dptr_, &beta,
                                     out_desc_, tmp.dptr_));
      mshadow::Copy(nodes_out[0]->data, tmp, nodes_out[0]->data.stream_);
    }

    virtual void Backprop(bool prop_grad,
                          const std::vector<Node<xpu>*> &nodes_in,
                          const std::vector<Node<xpu>*> &nodes_out,
                          ConnectState<xpu> *p_cstate) {
      mshadow::Tensor<xpu,4> &tmp = p_cstate->states[0];
      float alpha = 1.0f;
      float beta = 0.0f;
      if (prop_grad) {
        CUDA_CHECK(cudnnPoolingBackward(handle_, pooling_desc_, &alpha,
                                        out_desc_, tmp.dptr_,
                                        out_desc_, nodes_out[0]->data.dptr_,
                                        in_desc_, nodes_in[0]->data.dptr_, &beta,
                                        in_desc_, nodes_in[0]->data.dptr_));
      }
    }
  protected:
    inline void InitCuDNN() {
      init_cudnn_ = false;
      dtype_ = CUDNN_DATA_FLOAT;
      switch(mode) {
       case kMaxPooling: mode_ = CUDNN_POOLING_MAX; break;
       case kAvgPooling: mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING; break;
       default: utils::Error("This should not happen -,-"); break;
      }
      CUDA_CHECK(cudnnCreate(&handle_));
      CUDA_CHECK(cudnnCreateTensorDescriptor(&in_desc_));
      CUDA_CHECK(cudnnCreateTensorDescriptor(&out_desc_));
      CUDA_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc_));
      CUDA_CHECK(cudnnSetPooling2dDescriptor(pooling_desc_, mode_,
                                             Parent::param_.kernel_height,
                                             Parent::param_.kernel_width,
                                             0, 0,
                                             Parent::param_.stride,
                                             Parent::param_.stride));

    }
    /*! \brief cudnn init state flag*/
    bool init_cudnn_;
    /*! \brief cuDNN data type */
    cudnnDataType_t dtype_;
    /*! \brief cudnn handle */
    cudnnHandle_t handle_;
    /*! \brief cudnn pooling mode */
    cudnnPoolingMode_t mode_;
    /*! \brief input descriptor */
    cudnnTensorDescriptor_t in_desc_;
    /*! \brief output descriptor */
    cudnnTensorDescriptor_t out_desc_;
    /*! \brief pooling descriptor */
    cudnnPoolingDescriptor_t pooling_desc_;
#endif // CXXNET_USE_CUDNN
#endif // __CUDACC__
}; // class CuDNNPoolingLayer

}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_CUDNN_POOLING_LAYER_INL_HPP_

