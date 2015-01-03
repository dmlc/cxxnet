#ifndef CXXNET_LAYER_DROPOUT_LAYER_INL_HPP_
#define CXXNET_LAYER_DROPOUT_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./op.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class DropoutLayer : public ILayer<xpu> {
 public:
  DropoutLayer(mshadow::Random<xpu> *p_rnd) : prnd_(p_rnd) {
    // setup default value
    dropout_threshold = 0.0f;
  }
  virtual void SetParam(const char *name, const char* val) {
    if (!strcmp("threshold", name)) dropout_threshold = static_cast<real_t>(atof(val));
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "DropoutLayer: only support 1-1 connection");
    utils::Check(nodes_in[0] == nodes_out[0], "DropoutLayer is an self-loop Layer");
    utils::Check(dropout_threshold >= 0.0f && dropout_threshold < 1.0f,
                 "DropoutLayer: invalid dropout_threshold\n");
    // use 1 temp state for mask
    p_cstate->states.resize(1);
    p_cstate->states[0].Resize(nodes_in[0]->data.shape_);
  }
  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
                                  const std::vector<Node<xpu>*> &nodes_out,
                                  ConnectState<xpu> *p_cstate) {
    p_cstate->states[0].Resize(nodes_in[0]->data.shape_);  
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::TensorContainer<xpu,4> &mask = p_cstate->states[0];
    if (is_train) {
      const real_t pkeep = 1.0f - dropout_threshold;
      mask = F<op::threshold>(prnd_->uniform(mask.shape_), pkeep) * (1.0f / pkeep);
      nodes_out[0]->data = nodes_out[0]->data * mask;
    }
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::TensorContainer<xpu,4> &mask = p_cstate->states[0];
    if (prop_grad) {
      nodes_out[0]->data *= mask;
    }    
  }

 private:
  /*! \brief random number generator */
  mshadow::Random<xpu> *prnd_;
  /*! \brief dropout  */
  real_t dropout_threshold;
};  // class DropoutLayer
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_DROPOUT_LAYER_INL_HPP_

