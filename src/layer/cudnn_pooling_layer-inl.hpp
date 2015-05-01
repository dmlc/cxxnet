#ifndef CXXNET_LAYER_CUDNN_POOLING_LAYER_INL_HPP_
#define CXXNET_LAYER_CUDNN_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"

namespace cxxnet {
namespace layer {

template<typename Reducer, int mode, typename xpu>
class CuDNNPoolingLayer : public PoolingLayer<Reducer, mode, xpu> {
 public:
   CuDNNPoolingLayer(){}
};

#ifdef __CUDACC__
template<typename Reducer, int mode>
class CuDNNPoolingLayer<Reducer, mode, gpu> : public PoolingLayer<Reducer, mode, gpu> {
  private:
    typedef PoolingLayer<Reducer, mode, gpu> Parent;
  public:
    CuDNNPoolingLayer(){}
#if CXXNET_USE_CUDNN == 1
  public:
    virtual void InitConnection(const std::vector<Node<gpu>*> &nodes_in,
                                const std::vector<Node<gpu>*> &nodes_out,
                                ConnectState<gpu> *p_cstate) {
      Parent::InitNode(nodes_in, nodes_out, p_cstate);
      this->InitCuDNN();
      nodes_in[0]->must_contiguous = true;
      nodes_out[0]->must_contiguous = true;
    }
    virtual ~CuDNNPoolingLayer(void) {
      utils::Check(cudnnDestroyTensorDescriptor(in_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnDestroyTensorDescriptor(out_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnDestroyPoolingDescriptor(pooling_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnDestroy(handle_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
    }

    virtual void Forward(bool is_train,
                         const std::vector<Node<gpu>*> &nodes_in,
                         const std::vector<Node<gpu>*> &nodes_out,
                         ConnectState<gpu> *p_cstate) {
      mshadow::Tensor<gpu,4> &tmp = p_cstate->states[0];
      if (!init_cudnn_) {
        init_cudnn_ = true;
        utils::Check(cudnnSetStream(handle_, nodes_out[0]->data.stream_->stream_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
        mshadow::Tensor<gpu, 4, float> &in = nodes_in[0]->data;
        mshadow::Tensor<gpu, 4, float> &out = nodes_out[0]->data;
        utils::Check(cudnnSetTensor4dDescriptor(in_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                              in.shape_[0], in.shape_[1],
                                              in.shape_[2], in.shape_[3]) == CUDNN_STATUS_SUCCESS, "cudnn failed");
        utils::Check(cudnnSetTensor4dDescriptor(out_desc_, CUDNN_TENSOR_NCHW, dtype_,
                                              out.shape_[0], out.shape_[1],
                                              out.shape_[2], out.shape_[3]) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      }
      float alpha = 1.0f;
      float beta = 0.0f;

      CHECK(nodes_in[0]->data.CheckContiguous());
      CHECK(nodes_out[0]->data.CheckContiguous());
      CHECK(tmp.CheckContiguous());
      utils::Check(cudnnPoolingForward(handle_, pooling_desc_, &alpha,
                                     in_desc_, nodes_in[0]->data.dptr_, &beta,
                                     out_desc_, tmp.dptr_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      mshadow::Copy(nodes_out[0]->data, tmp, nodes_out[0]->data.stream_);
    }
    virtual void Backprop(bool prop_grad,
                          const std::vector<Node<gpu>*> &nodes_in,
                          const std::vector<Node<gpu>*> &nodes_out,
                          ConnectState<gpu> *p_cstate) {
      mshadow::Tensor<gpu,4> &tmp = p_cstate->states[0];
      mshadow::Tensor<gpu,4> &tmp2 = p_cstate->states[1];
      float alpha = 1.0f;
      float beta = 0.0f;
      if (prop_grad) {
        utils::Check(cudnnPoolingBackward(handle_, pooling_desc_, &alpha,
                                        out_desc_, tmp.dptr_,
                                        out_desc_, nodes_out[0]->data.dptr_,
                                        in_desc_, nodes_in[0]->data.dptr_, &beta,
                                        in_desc_, tmp2.dptr_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      }
      mshadow::Copy(nodes_in[0]->data, tmp2, nodes_in[0]->data.stream_);
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
      utils::Check(cudnnCreate(&handle_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnCreateTensorDescriptor(&in_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnCreateTensorDescriptor(&out_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnCreatePoolingDescriptor(&pooling_desc_) == CUDNN_STATUS_SUCCESS, "cudnn failed");
      utils::Check(cudnnSetPooling2dDescriptor(pooling_desc_, mode_,
                                             Parent::param_.kernel_height,
                                             Parent::param_.kernel_width,
                                             Parent::param_.pad_y, Parent::param_.pad_x,
                                             Parent::param_.stride,
                                             Parent::param_.stride) == CUDNN_STATUS_SUCCESS, "cudnn failed");

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
}; // class CuDNNPoolingLayer
#endif // __CUDACC__
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_CUDNN_POOLING_LAYER_INL_HPP_

