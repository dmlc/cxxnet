#ifndef CXXNET_LAYER_POOLING_LAYER_INL_HPP_
#define CXXMET_LAYER_POOLING_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"

namespace cxxnet {
namespace layer {

template<typename Reducer, bool scalebysize, typename xpu>
class PoolingLayer : public ILayer<xpu> {
 public:
  virtual ~PoolingLayer(void) {}
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam( name, val );
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "PoolingLayer: only support 1-1 connection");
    const index_t ksize_y = static_cast<index_t>(param_.kernel_height);
    const index_t ksize_x = static_cast<index_t>(param_.kernel_width);
    const index_t kstride = static_cast<index_t>(param_.stride);
    mshadow::Shape<4> ishape = nodes_in[0]->data.shape;
    utils::Check(param_.kernel_height > 0 && param_.kernel_width > 0, "must set kernel_size correctly" );
    utils::Check(ksize_x <= ishape[0] && ksize_y <= ishape[1], "kernel size exceed input" );
    
    mshadow::Shape<4> oshape = mshadow::
        Shape4( ishape[3], ishape[2],
                std::min(ishape[1] - ksize_y + kstride-1, ishape[1] - 1) / kstride + 1,
                std::min(ishape[0] - ksize_x + kstride-1, ishape[0] - 1) / kstride + 1);
    nodes_out[0]->data.shape = oshape;
    // use 1 temp state to store pooled result
    p_cstate->states.resize(1);    
    p_cstate->states[0].Resize(oshape);
  }
  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
                                  const std::vector<Node<xpu>*> &nodes_out,
                                  ConnectState<xpu> *p_cstate) {
    p_cstate->states[0].Resize(nodes_out[0]->data.shape);
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu,4> &tmp = p_cstate->states[0];
    const int ksize_y = param_.kernel_height;
    const int ksize_x = param_.kernel_width;
    mshadow::Shape<2> pshape = nodes_out[0]->data[0][0].shape;
    if (!scalebysize) {
      tmp = pool<Reducer>(nodes_in[0]->data, pshape, ksize_y, ksize_x, param_.stride);
    }else{
      tmp = pool<Reducer>(nodes_in[0]->data, pshape, ksize_y, ksize_x, param_.stride) 
          * (1.0f / (ksize_y*ksize_x));
    }
    mshadow::Copy(nodes_out[0]->data, tmp);
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
      if (!scalebysize) {
        nodes_in[0]->data = unpool<Reducer>(nodes_in[0]->data, tmp, nodes_out[0]->data, ksize_y, ksize_x, param_.stride);
      }else{
        nodes_in[0]->data = unpool<Reducer>(nodes_in[0]->data, tmp, nodes_out[0]->data, ksize_y, ksize_x, param_.stride)
            * (1.0f / (ksize_y * ksize_x));
      }
    }
  }

 private:
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
};   // class PoolingLayer
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_POOLING_LAYER_INL_HPP_

