#ifndef LAYER_LRN_LAYER_INL_HPP_
#define LAYER_LRN_LAYER_INL_HPP_

#include "./layer.h"
#include "./op.h"

namespace cxxnet {
namespace layer {

template<typename xpu>
class LRNLayer : public CommonLayerBase<xpu> {
 public:
  LRNLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : CommonLayerBase<xpu>(p_rnd, p_in, p_out) {
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

 protected:
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) {
    pnode_out->data.shape = node_in.data.shape;
    tmp_in.Resize(node_in.data.shape);
    tmp_norm.Resize(node_in.data.shape);
  }
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    using namespace mshadow;
    using namespace mshadow::expr;
    const real_t salpha = alpha_ / nsize_;
    // stores normalizer without power
    tmp_norm = chpool<red::sum>(F<op::square>(pnode_in->data) , nsize_) * salpha + knorm_;
    pnode_out->data = pnode_in->data * F<op::power>(tmp_norm, -beta_);
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    using namespace mshadow;
    using namespace mshadow::expr;
    const real_t salpha = alpha_ / nsize_;
    if(prop_grad) {
      // backup input data
      mshadow::Copy(tmp_in, pnode_in->data);
      // first gradient to a[i], will be 1 / normalizer
      pnode_in->data = pnode_out->data * F<op::power>(tmp_norm, -beta_);
      // gradient to normalizer
      pnode_in->data += (- 2.0f * beta_ * salpha) * 
          chpool<red::sum>(pnode_out->data * tmp_in * F<op::power>(tmp_norm, -beta_-1.0f), nsize_)  * tmp_in;
    }
  }
 private:
  /*! \brief input temp data */
  mshadow::TensorContainer<xpu,4> tmp_in;
  /*! \brief temp normalizer */
  mshadow::TensorContainer<xpu,4> tmp_norm;
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

