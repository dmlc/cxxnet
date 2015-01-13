#ifndef CXXNET_LAYER_CUDNN_POOLING_LAYER_INL_HPP_
#define CXXNET_LAYER_CUDNN_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include <cudnn.h>

namespace cxxnet {
namespace layer {
template<typename Reducer, bool scalebysize, typename xpu>
class CuDNNPoolingLayer : public ILayer<xpu> {
 public:
  virtual ~CuDNNPoolingLayer(void) {}
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam(name, val);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    CUDA_CHECK(cudnnCreate(&handle_));
    CUDA_CHECK(cudnnCreateTensorDescriptor(&in_desc_));
    CUDA_CHECK(cudnnCreateTensorDescriptor(&out_desc_));
    CUDA_CHECK(cudnnCreatePoolingDescriptor(&pooling_desc_));

    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "PoolingLayer: only support 1-1 connection");
    const index_t ksize_y = static_cast<index_t>(param_.kernel_height);
    const index_t ksize_x = static_cast<index_t>(param_.kernel_width);
    const index_t kstride = static_cast<index_t>(param_.stride);
    mshadow::Shape<4> ishape = nodes_in[0]->data.shape_;
    utils::Check(param_.kernel_height > 0 && param_.kernel_width > 0, "must set kernel_size correctly");
    utils::Check(ksize_x <= ishape[3] && ksize_y <= ishape[2], "kernel size exceed input");

    mshadow::Shape<4> oshape = mshadow::
        Shape4(ishape[0], ishape[1],
               std::min(ishape[2] - ksize_y + kstride-1, ishape[2] - 1) / kstride + 1,
               std::min(ishape[3] - ksize_x + kstride-1, ishape[3] - 1) / kstride + 1);
    nodes_out[0]->data.shape_ = oshape;
    // use 1 temp state to store pooled result
    init_cudnn_ = false;
    alpha_ = 1.0f;
    beta_ = 0.0f;
    CUDA_CHECK(cudnnSetPooling2dDescriptor(pooling_desc_, CUDNN_POOLING_MAX,
                                ksize_y, ksize_x, 0, 0, kstride, kstride));
    p_cstate->states.resize(1);
    p_cstate->states[0].Resize(oshape);
  }
  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
                                  const std::vector<Node<xpu>*> &nodes_out,
                                  ConnectState<xpu> *p_cstate) {
    p_cstate->states[0].Resize(nodes_out[0]->data.shape_);
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    mshadow::Tensor<xpu,4> &tmp = p_cstate->states[0];
    const index_t ksize_y = static_cast<index_t>(param_.kernel_height);
    const index_t ksize_x = static_cast<index_t>(param_.kernel_width);
    if (!init_cudnn_) {
      init_cudnn_ = true;
      mshadow::Tensor<gpu, 4, float> &sgt = nodes_in[0]->data;
      mshadow::Tensor<gpu, 4, float> &dgt = tmp;
      CUDA_CHECK(cudnnSetTensor4dDescriptorEx(in_desc_,CUDNN_DATA_FLOAT,
                                   sgt.shape_[0], sgt.shape_[1], sgt.shape_[2], sgt.shape_[3],
                                   sgt.stride_ * sgt.shape_[1] * sgt.shape_[2],
                                   sgt.stride_ * sgt.shape_[2],
                                   sgt.stride_,
                                   1));
      CUDA_CHECK(cudnnSetTensor4dDescriptorEx(out_desc_, CUDNN_DATA_FLOAT,
                                   dgt.shape_[0], dgt.shape_[1], dgt.shape_[2], dgt.shape_[3],
                                   dgt.stride_ * dgt.shape_[1] * dgt.shape_[2],
                                   dgt.stride_ * dgt.shape_[2],
                                   dgt.stride_,
                                   1));

    }
    CUDA_CHECK(cudnnPoolingForward(handle_, pooling_desc_, &alpha_,
                                          in_desc_, nodes_in[0]->data.dptr_, &beta_,
                                          out_desc_, tmp.dptr_));
    if (scalebysize) {
      tmp *= 1.0f / (ksize_y*ksize_x);
    }
    mshadow::Copy(nodes_out[0]->data, tmp, nodes_out[0]->data.stream_);
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu,4> &tmp = p_cstate->states[0];
    if (prop_grad) {
      const int ksize_y = param_.kernel_height;
      const int ksize_x = param_.kernel_width;
      CUDA_CHECK(cudnnPoolingBackward(handle_, pooling_desc_, &alpha_,
                           out_desc_, tmp.dptr_, out_desc_, nodes_out[0]->data.dptr_,
                           in_desc_, nodes_in[0]->data.dptr_, &beta_,
                           in_desc_, nodes_in[0]->data.dptr_));
      if (scalebysize) {
        nodes_in[0]->data *=  (1.0f / (ksize_y * ksize_x));
      }
    }
  }

 private:
  /*! \brief cudnn init state flag*/
  bool init_cudnn_;
  /*! \brief alpha for cudnn */
  float alpha_;
  /*! \brief beta for cudnn */
  float beta_;
  /*! \brief parameters are potentially useful */
  LayerParam param_;
  /*! \brief cudnn handle */
  cudnnHandle_t handle_;
  /*! \brief input descriptor */
  cudnnTensorDescriptor_t in_desc_;
  /*! \brief output descriptor */
  cudnnTensorDescriptor_t out_desc_;
  /*! \brief pooling descriptor */
  cudnnPoolingDescriptor_t pooling_desc_;
};   // class PoolingLayer
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_POOLING_LAYER_INL_HPP_

