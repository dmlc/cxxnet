#include "./layer.h"
#include "./param.h"
#include "mshadow/tensor_container.h"

namespace cxxnet {
namespace layer {
template<typename xpu>
class FullConnectLayer : public CommonLayerBase<xpu> {
 public:
  FullConnectLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : CommonLayerBase<xpu>(p_rnd, p_in, p_out) {}
  virtual ~FullConnectLayer(void) {}
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam(name, val);
  }
  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit("wmat", wmat_, gwmat_);
    if (param_.no_bias == 0) {
      pvisitor->Visit("bias", bias_, gbias_);
    }
  }
  virtual void InitModel(void) {
    // rexsize to correct shape
    wmat_.Resize(mshadow::Shape2(this->pout_->data.shape[0], this->pin_->data.shape[0]));
    bias_.Resize(mshadow::Shape1(this->pout_->data.shape[0]));
    param_.RandInitWeight(this->prnd_, wmat_, wmat_.shape[0], wmat_.shape[1]);
    bias_ = param_.init_bias;
    // setup gradient weight
    gwmat_.Resize(wmat_.shape);
    gbias_.Resize(bias_.shape);
    gwmat_ = 0.0f; gbias_ = 0.0f;
  }
  virtual void SaveModel(mshadow::utils::IStream &fo) const{
    fo.Write(&param_, sizeof(LayerParam));
    wmat_.SaveBinary(fo);
    bias_.SaveBinary(fo);
  }
  virtual void LoadModel(mshadow::utils::IStream &fi){
    utils::Assert(fi.Read(&param_, sizeof(LayerParam)) != 0,
                  "FullConnectLayer:LoadModel invalid model file");
    wmat_.LoadBinary(fi);
    bias_.LoadBinary(fi);
    // setup gradient weight
    gwmat_.Resize(wmat_.shape);
    gbias_.Resize(bias_.shape);
    gwmat_ = 0.0f; gbias_ = 0.0f;
  }
 protected:
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) {
    utils::Assert(node_in.is_mat(), "input need to be a matrix");
    utils::Check(param_.num_hidden > 0, "must set nhidden correctly");
    // we change matrix convention 
    pnode_out->data.shape = mshadow::Shape4(node_in.data.shape[4], 1, 1, param_.num_hidden);
  }
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    mshadow::Tensor<xpu, 2> m_in = pnode_in->mat();
    mshadow::Tensor<xpu, 2> m_out = pnode_out->mat();
    index_t nbatch = m_in.shape[1];
    m_out = dot(m_in, wmat_.T());
    if (param_.no_bias == 0) {
      m_out += repmat(bias_, nbatch);
    }
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    mshadow::Tensor<xpu, 2> m_in = pnode_in->mat();
    mshadow::Tensor<xpu, 2> m_out = pnode_out->mat();
    // accumulate gradient
    gwmat_ += dot(m_out.T(), m_in);
    if (param_.no_bias == 0) {
      gbias_ += sum_rows(m_out);
    }
    // backprop
    if (prop_grad) {
      m_in = dot(m_out, wmat_);
    }
  }
 protected:
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
  /*! \brief weight matrix */
  mshadow::TensorContainer<xpu,2> wmat_;
  /*! \brief bias */
  mshadow::TensorContainer<xpu,1> bias_;
  /*! \brief accumulates the gradient of weight matrix */
  mshadow::TensorContainer<xpu,2> gwmat_;
  /*! \brief accumulates the gradient of bias */
  mshadow::TensorContainer<xpu,1> gbias_;
};
}
}
