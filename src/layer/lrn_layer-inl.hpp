#ifndef CXXNET_LAYER_LRN_LAYER_INL_HPP_
#define CXXNET_LAYER_LRN_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./op.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class LRNLayer : public ILayer<xpu> {
 public:
  LRNLayer(void) {
    // default values
    this->knorm_ = 1.0f;
    this->nsize_ = 3;
  }
  virtual ~LRNLayer(void){}
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "local_size")) nsize_ = static_cast<index_t>(atoi(val));
    if (!strcmp(name, "alpha")) alpha_ = static_cast<real_t>(atof(val));
    if (!strcmp(name, "beta")) beta_ = static_cast<real_t>(atof(val));
    if (!strcmp(name, "knorm")) knorm_ = static_cast<real_t>(atof(val));
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    tmp_in.set_stream(stream);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "LRNLayer: only support 1-1 connection");
    nodes_out[0]->data.shape_ = nodes_in[0]->data.shape_;
    // use 1 temp state for mask
    p_cstate->states.resize(1);
    p_cstate->states[0].Resize(nodes_in[0]->data.shape_);
    // temp in is kepted in layer, since it does not go across forward/backprop
    tmp_in.Resize(nodes_in[0]->data.shape_);
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
    using namespace mshadow;
    using namespace mshadow::expr;
    mshadow::Tensor<xpu,4> &tmp_norm = p_cstate->states[0];
    const real_t salpha = alpha_ / nsize_;
    // stores normalizer without power
    tmp_norm = chpool<red::sum>(F<op::square>(nodes_in[0]->data) , nsize_) * salpha + knorm_;
    nodes_out[0]->data = nodes_in[0]->data * F<op::power>(tmp_norm, -beta_);
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow;
    using namespace mshadow::expr;
    tmp_in.Resize(nodes_in[0]->data.shape_);
    mshadow::Tensor<xpu,4> &tmp_norm = p_cstate->states[0];
    const real_t salpha = alpha_ / nsize_;
    if (prop_grad) {
      // backup input data
      mshadow::Copy(tmp_in, nodes_in[0]->data, tmp_in.stream_);
      // first gradient to a[i], will be 1 / normalizer
      nodes_in[0]->data = nodes_out[0]->data * F<op::power>(tmp_norm, -beta_);
      // gradient to normalizer
      nodes_in[0]->data += (- 2.0f * beta_ * salpha) * 
          chpool<red::sum>(nodes_out[0]->data * tmp_in * F<op::power>(tmp_norm, -beta_-1.0f), nsize_)  * tmp_in;      
    }
  }
  
 private:
  /*! \brief input temp data */
  mshadow::TensorContainer<xpu,4> tmp_in;
  /*! \brief alpha */
  real_t alpha_;
  /*! \brief beta */
  real_t beta_;
  /*! \brief knorm */
  real_t knorm_;
  /*! \brief neighbor size */
  index_t nsize_;
}; // class lrn layer
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_LRN_LAYER_INL_HPP_

