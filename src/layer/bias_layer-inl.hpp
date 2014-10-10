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
class BiasLayer : public ILayer<xpu> {
 public:
  virtual ~BiasLayer( void ){}
  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit("bias", bias_, gbias_);
  }
  virtual void SetParam(const char *name, const char* val){
    param_.SetParam(name, val);
  }
  virtual void InitModel(void) {
    bias_.Resize(mshadow::Shape1(param_.num_input_node));
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

  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "BiasLayer Layer only support 1-1 connection");
    utils::Check(nodes_in[0] == nodes_out[0], "BiasLayer is an self-loop Layer");
    utils::Check(nodes_in[0]->is_mat(), "BiasLayer only works for flatten node so far");
    if (param_.num_input_node == 0) {
      param_.num_input_node = static_cast<int>(nodes_in[0]->data.shape[0]);
    } else {
      utils::Check(param_.num_input_node == static_cast<int>(nodes_in[0]->data.shape[0]),
                   "BiasLayer: input hidden nodes is not consistent");
    }
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    // InitConnection is already called, no need to check shape again
    mshadow::index_t nbatch = nodes_in[0]->data.shape[1];
    nodes_in[0]->mat() += repmat(bias_, nbatch);
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    gbias_ += sum_rows(nodes_in[0]->mat());
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

