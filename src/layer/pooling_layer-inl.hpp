#ifndef CXXNET_LAYER_POOLING_LAYER_INL_HPP_
#define CXXNET_LAYER_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"

namespace cxxnet {
namespace layer {

template<typename Reducer,
         int mode,
         typename xpu,
         bool is_identity = true,
         typename ForwardOp = op::identity,
         typename BackOp = op::identity_grad>
class PoolingLayer : public ILayer<xpu> {
 public:
  virtual ~PoolingLayer(void) {}
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam(name, val);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    InitNode(nodes_in, nodes_out, p_cstate);
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
    using namespace mshadow::expr;
    mshadow::Tensor<xpu,4> &tmp = p_cstate->states[0];
    const int ksize_y = param_.kernel_height;
    const int ksize_x = param_.kernel_width;
    mshadow::Shape<2> pshape = nodes_out[0]->data[0][0].shape_;
    if (!is_identity) {
      nodes_in[0]->data = F<ForwardOp>(nodes_in[0]->data);
    }
    if (mode == kMaxPooling || mode == kSumPooling) {
      tmp = pool<Reducer>(nodes_in[0]->data, pshape, ksize_y, ksize_x, param_.stride);
    }else if (mode == kAvgPooling) {
      tmp = pool<Reducer>(nodes_in[0]->data, pshape, ksize_y, ksize_x, param_.stride)
          * (1.0f / (ksize_y*ksize_x));
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
      if (is_identity) {
        if (mode == kMaxPooling || mode == kSumPooling) {
          nodes_in[0]->data = unpool<Reducer>(nodes_in[0]->data, tmp, nodes_out[0]->data, ksize_y, ksize_x, param_.stride);
        }else if (mode == kAvgPooling) {
          nodes_in[0]->data = unpool<Reducer>(nodes_in[0]->data, tmp, nodes_out[0]->data, ksize_y, ksize_x, param_.stride)
              * (1.0f / (ksize_y * ksize_x));
        }
      }  else {
        if (mode == kMaxPooling || mode == kSumPooling) {
          nodes_in[0]->data = F<BackOp>(nodes_in[0]->data) *
              unpool<Reducer>(nodes_in[0]->data, tmp, nodes_out[0]->data, ksize_y, ksize_x, param_.stride);
        } else if (mode == kAvgPooling) {
          nodes_in[0]->data = F<BackOp>(nodes_in[0]->data) *
              unpool<Reducer>(nodes_in[0]->data, tmp, nodes_out[0]->data, ksize_y, ksize_x, param_.stride)
              * (1.0f / (ksize_y * ksize_x));
        }
      }
    }
  }

 protected:
  inline void InitNode(const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "PoolingLayer: only support 1-1 connection");
    const index_t ksize_y = static_cast<index_t>(param_.kernel_height);
    const index_t ksize_x = static_cast<index_t>(param_.kernel_width);
    const index_t kstride = static_cast<index_t>(param_.stride);
    mshadow::Shape<4> ishape = nodes_in[0]->data.shape_;
    utils::Check(param_.kernel_height > 0 && param_.kernel_width > 0,
                 "must set kernel_size correctly");
    utils::Check(ksize_x <= ishape[3] && ksize_y <= ishape[2],
                 "kernel size exceed input");

    mshadow::Shape<4> oshape = mshadow::
        Shape4(ishape[0], ishape[1],
               std::min(ishape[2] - ksize_y + kstride-1, ishape[2] - 1) / kstride + 1,
               std::min(ishape[3] - ksize_x + kstride-1, ishape[3] - 1) / kstride + 1);
    nodes_out[0]->data.shape_ = oshape;
    // use 1 temp state to store pooled result
    p_cstate->states.resize(1);
    p_cstate->states[0].Resize(oshape);
  }
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
};   // class PoolingLayer
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_POOLING_LAYER_INL_HPP_

