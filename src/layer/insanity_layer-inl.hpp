#ifndef INSANITY_LAYER_INL_HPP
#define INSANITY_LAYER_INL_HPP
#pragma once

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./op.h"


namespace cxxnet {
namespace layer {

template<typename xpu>
class InsanityLayer : public ILayer<xpu> {
 public:
  InsanityLayer(mshadow::Random<xpu> *p_rnd) : prnd_(p_rnd) {
    // setup default value
    lb_ = 5.0f;
    ub_ = 10.0f;
    step_ = 0;
    saturation_start_ = 0;
    saturation_end_ = 0;
    delta_ = 0.0f;
    init_ = false;
  }
  virtual void SetParam(const char *name, const char* val) {
    if (!strcmp("lb", name)) lb_ = atof(val);
    if (!strcmp("ub", name)) ub_ = atof(val);
    if (!strcmp("calm_start", name)) saturation_start_ = atol(val);
    if (!strcmp("calm_end",  name)) saturation_end_ = atol(val);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "InsanityLayer: only support 1-1 connection");
    // use 1 temp state for mask
    p_cstate->states.resize(1);
    p_cstate->states[0].Resize(nodes_in[0]->data.shape_);
    nodes_out[0]->data.shape_ = nodes_in[0]->data.shape_;
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
    if (!init_) {
      init_ = true;
      delta_ = (ub_ - lb_) / (log(ub_) - log(lb_));
      delta_ = ub_ - delta_;
      delta_ /= (saturation_end_ - saturation_start_);
    }
    using namespace mshadow::expr;
    if (step_ < saturation_end_ && step_ > saturation_start_) {
      ub_ -= delta_ * step_;
      lb_ += delta_ * step_;
      step_ ++;
    }
    mshadow::TensorContainer<xpu,4> &mask = p_cstate->states[0];
    if (is_train) {
      mask = prnd_->uniform(mask.shape_);
      mask = mask * (ub_ - lb_) + lb_;
      nodes_in[0]->data = F<op::xelu>(nodes_in[0]->data, mask);
      mshadow::Copy(nodes_out[0]->data, nodes_in[0]->data, nodes_out[0]->data.stream_);
    } else {
      nodes_in[0]->data = F<op::xelu>(nodes_in[0]->data, (ub_ - lb_) / (log(ub_) - log(lb_)));
      mshadow::Copy(nodes_out[0]->data, nodes_in[0]->data, nodes_out[0]->data.stream_);
    }
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::TensorContainer<xpu,4> &mask = p_cstate->states[0];
    if (prop_grad) {
      nodes_in[0]->data = F<op::xelu_grad>(nodes_in[0]->data, mask) * nodes_out[0]->data;
    }
  }

 private:
  /*! \brief random number generator */
  mshadow::Random<xpu> *prnd_;
  /*! \brief whether initialized */
  bool init_;
  /*! \brief lower bound */
  float lb_;
  /*! \brief upper bound */
  float ub_;
  /*! \brief step counter */
  long step_;
  /*! \brief epoch to start saturation process */
  long saturation_start_;
  /*! \brief epoch to finish saturation */
  long saturation_end_;
  /*! \brief change in each epoch */
  double delta_;
};  // class InsanityLayer

} // namespace layer
} // namespace cxxnet
#endif // INSANITY_LAYER_INL_HPP
