#include "./layer.h"
#include "./op.h"
#include "mshadow/tensor_container.h"

namespace cxxnet {
namespace layer {
template<typename xpu>
class DropoutLayer : public CommonLayerBase<xpu> {
 public:
  DropoutLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : CommonLayerBase<xpu>(p_rnd, p_in, p_out) {
    utils::Check(p_in == p_out, "dropout layer must self loop e.g layer[1->1] = dropout");
    // setup default value
    dropout_threshold = 0.0f;
  }
  virtual void SetParam(const char *name, const char* val) {
    if (!strcmp("threshold", name)) dropout_threshold = static_cast<real_t>(atof(val));
  }

 protected:
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) {
    utils::Check(dropout_threshold >= 0.0f && dropout_threshold < 1.0f,
                 "DropoutLayer: invalid dropout_threshold\n");
    mask_.Resize(node_in.data.shape);
  }
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    if (is_train) {
      const real_t pkeep = 1.0f - dropout_threshold;
      mask_ = F<op::threshold>(this->prnd_->uniform(mask_.shape), pkeep) * (1.0f / pkeep);
      pnode_out->data = pnode_out->data * mask_;
    }
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    if (prop_grad) {
      pnode_out->data *= mask_;
    }
  }

 private:
  /*! \brief dropout mask */
  mshadow::TensorContainer<xpu, 4> mask_;
  /*! \brief dropout  */
  real_t dropout_threshold;
};  // class DropoutLayer
}  // namespace layer
}  // namespace cxxnet
