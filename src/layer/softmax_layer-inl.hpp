#ifndef CXXNET_LAYER_SOFTMAX_LAYER_INL_HPP_
#define CXXNET_LAYER_SOFTMAX_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"

namespace cxxnet {
namespace layer {
/*! \brief softmax layer, do softmax transformation during forward */
template<typename xpu>
class SoftmaxLayer: public ILayer<xpu> {
 public:
  SoftmaxLayer(const std::vector<Node<xpu>*> &pnode_in, 
               const std::vector<Node<xpu>*> &pnode_out,                  
               const LabelInfo *label_info) {
    utils::Check(pnode_in.size() == 1 && pnode_out.size() == 1 && pnode_in[0] == pnode_out[0],
                 "softmax layer must self loop e.g layer[1->1] = softmax");
    this->pin = pnode_in[0];
    this->plabelinfo = label_info;    
  }
  virtual ~SoftmaxLayer(void) {}
  virtual void InitLayer(void) {
    tnode.Resize(pin->mat().shape);
  }
  virtual void Forward(bool is_train) {
    mshadow::Softmax(pin->mat(), pin->mat());
  }
  virtual void Backprop(bool prop_grad) {
    // do computation in CPU for simplicity, since this is not bottle neck
    mshadow::Copy(tnode, pin->mat());
    for (mshadow::index_t i = 0; i < plabelinfo->batch_size; ++i) {
      index_t k = static_cast<index_t>(plabelinfo->labels[i]);
      tnode[i][k] -= 1.0f;
    }
    mshadow::Copy(pin->mat(), tnode);
  }
  
 protected:
    
  /*! \brief temp space for notde*/
  mshadow::TensorContainer<cpu,2> tnode;
  /*! \brief reference to label information */
  const LabelInfo *plabelinfo;
  /*! \brief input and output node type */
  Node<xpu> *pin;
};
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_SOFTMAX_LAYER_INL_HPP_
