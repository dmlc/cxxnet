#ifndef CXXNET_LAYER_BIAS_LAYER_INL_HPP_
#define CXXNET_LAYER_BIAS_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {

/*! \brief a simple layer that adds bias to every node in batch, this is a self-loop layer */
template<typename xpu>
class BiasLayer : public CommonLayerBase<xpu> {
 public:
  BiasLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : CommonLayerBase<xpu>(p_rnd, p_in, p_out) {
    utils::Check(p_in == p_out, "bias layer must self loop e.g layer[1->1] = dropout");
  }
  virtual ~BiasLayer( void ){}
  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit("bias", bias_, gbias_);
  }
  virtual void SetParam(const char *name, const char* val){
    param_.SetParam(name, val);
  }
  virtual void InitModel(void) {
    bias_.Resize(mshadow::Shape1(this->pin_->data.shape[0]));
    bias_ = param_.init_bias;
    gbias_.Resize(bias_.shape);
    gbias_ = 0.0f;
  }
  virtual void SaveModel(utils::IStream &fo) const{
    fo.Write(&param_, sizeof(LayerParam));
    bias_.SaveBinary(fo);
  }
  virtual void LoadModel(utils::IStream &fi){
    utils::Check(fi.Read(&param_, sizeof(LayerParam) ) != 0,
                 "BiasLayer: LoadModel invalid model file");
    bias_.LoadBinary(fi);
    gbias_.Resize(bias_.shape);
    gbias_ = 0.0f;
  }

 protected:
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) {
    utils::Assert(node_in.is_mat(), "bias layer only works for flatten node so far");
  }
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    mshadow::index_t nbatch = pnode_in->data.shape[1];
    pnode_in->mat() += repmat(bias_, nbatch);
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    gbias_ += sum_rows(pnode_in->mat());            
  }

 private:
  /*! \brief parameters that potentially be useful */
  LayerParam param_; 
  /*! \brief bias */
  mshadow::TensorContainer<xpu,1> bias_;
  /*! \brief accumulates the gradient of bias */
  mshadow::TensorContainer<xpu,1> gbias_;
};
}  // namespace layer
}  // namespace cxxnet
#endif // LAYER_BIAS_LAYER_INL_HPP_

