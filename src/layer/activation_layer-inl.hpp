#include "./layer.h"
#include "./param.h"
namespace cxxnet {
namespace layer {
using namespace mshadow::expr;
template<typename xpu,typename ForwardOp, typename BackOp>
class ActivationLayer : public CommonLayerBase<xpu>{
 public:
  ActivationLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : CommonLayerBase<xpu>(p_rnd, p_in, p_out) {}
  virtual ~ActivationLayer(void) {}
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) {
    pnode_out->data.shape = node_in.data.shape;
  }
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    pnode_in->data = F<ForwardOp>(pnode_in->data);
    mshadow::Copy(pnode_out->data, pnode_in->data);
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    if (prop_grad) {
      pnode_in->data = F<BackOp>(pnode_in->data) * pnode_out->data;
    }
  }
};
}  // namespace layer

/*! \brief operations for ActivationLayer */
namespace op {
/*! \brief sigmoid unit */
struct sigmoid {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f / (1.0f + expf(-a));
  }
};
struct sigmoid_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a * (1.0f - a);
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    using namespace std;
    return max(a, 0.0f);
  }
};
struct relu_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return a > 0.0f ? 1.0f : 0.0f;
  }
};

struct tanh {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return tanhf( a );
  }
};

struct tanh_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f - a * a;
  }
};

struct softplus {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return logf(1 + expf(a));
  }
};

struct softplus_grad {
  MSHADOW_XINLINE static real_t Map(real_t a) {
    return 1.0f / (1.0f + expf(-a));
  }
};

}  // namespace op
}  // namespace cxxnet
