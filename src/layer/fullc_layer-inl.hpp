#ifndef CXXNET_LAYER_FULLC_LAYER_INL_HPP_
#define CXXNET_LAYER_FULLC_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "./op.h"
#include "../utils/utils.h"

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
  virtual void SaveModel(utils::IStream &fo) const{
    fo.Write(&param_, sizeof(LayerParam));
    wmat_.SaveBinary(fo);
    bias_.SaveBinary(fo);
  }
  virtual void LoadModel(utils::IStream &fi){
    utils::Check(fi.Read(&param_, sizeof(LayerParam)) != 0,
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
    utils::Check(node_in.is_mat(), "input need to be a matrix");
    utils::Check(param_.num_hidden > 0, "must set nhidden correctly");
    // we change matrix convention 
    pnode_out->data.shape = mshadow::Shape4(node_in.data.shape[3], 1, 1, param_.num_hidden);
  }
  
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    this->Forward_(is_train, wmat_, pnode_in, pnode_out);
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    this->Backprop_(prop_grad, wmat_, pnode_in, pnode_out);
  }
  // internal implementation
  inline void Forward_(bool is_train,
                       mshadow::Tensor<xpu,2> wmat,                        
                       Node<xpu> *pnode_in,
                       Node<xpu> *pnode_out) {
    mshadow::Tensor<xpu, 2> m_in = pnode_in->mat();
    mshadow::Tensor<xpu, 2> m_out = pnode_out->mat();
    index_t nbatch = m_in.shape[1];
    m_out = dot(m_in, wmat.T());
    if (param_.no_bias == 0) {
      m_out += repmat(bias_, nbatch);
    }
  }
  inline void Backprop_(bool prop_grad,
                        mshadow::Tensor<xpu,2> wmat,
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
      m_in = dot(m_out, wmat);
    }
  }

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

// dropconn layer that randomly drops connection in fullc
template<typename xpu>
class DropConnLayer : public FullConnectLayer<xpu> {
 public:
  DropConnLayer(mshadow::Random<xpu> *p_rnd, Node<xpu> *p_in, Node<xpu> *p_out)
      : Parent(p_rnd, p_in, p_out) {
    dropout_threshold = 0.0f;
  }
  virtual void SetParam(const char *name, const char* val) {
    Parent::SetParam(name, val);
    if (!strcmp("threshold", name)) dropout_threshold = static_cast<real_t>(atof(val));
  }  

 protected:
  virtual void InitLayer_(const Node<xpu> &node_in,
                          Node<xpu> *pnode_out) {
    Parent::InitLayer_(node_in, pnode_out);
    this->mask_.Resize(mshadow::Shape2(this->pin_->data.shape[0], this->pin_->data.shape[0]));
  }
  
  virtual void Forward_(bool is_train,
                        Node<xpu> *pnode_in,
                        Node<xpu> *pnode_out) {
    using namespace mshadow::expr;
    if (is_train) {
      const real_t pkeep = 1.0f - dropout_threshold;
      mask_ = F<op::threshold>(this->prnd_->uniform(mask_.shape), pkeep) * (1.0f / pkeep);
      tmpw_ = this->wmat_ * mask_;
    } else {
      mshadow::Copy(tmpw_, this->wmat_);
    }
    Parent::Forward_(is_train, tmpw_, pnode_in, pnode_out);
  }
  virtual void Backprop_(bool prop_grad,
                         Node<xpu> *pnode_in,
                         Node<xpu> *pnode_out) {
    Parent::Backprop_(prop_grad, tmpw_, pnode_in, pnode_out);
    Parent::gwmat_ *= mask_;
  }

 private:
  mshadow::real_t dropout_threshold;
  typedef FullConnectLayer<xpu> Parent;
  mshadow::TensorContainer<xpu, 2> mask_, tmpw_;
};  // class DropconnLayer
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_FULLC_LAYER_INL_HPP_
