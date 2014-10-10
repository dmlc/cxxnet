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
class FullConnectLayer : public ILayer<xpu> {
 public:
  FullConnectLayer(mshadow::Random<xpu> *p_rnd) : prnd_(p_rnd) {}
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
    wmat_.Resize(mshadow::Shape2(param_.num_hidden, param_.num_input_node));
    bias_.Resize(mshadow::Shape1(param_.num_hidden));
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

  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "DropoutLayer Layer only support 1-1 connection");
    utils::Check(nodes_in[0]->is_mat(), "input need to be a matrix");
    utils::Check(param_.num_hidden > 0, "must set nhidden correctly");
    // we change matrix convention 
    nodes_out[0]->data.shape = 
        mshadow::Shape4(nodes_in[0]->data.shape[3], 1, 1, param_.num_hidden);
    if (param_.num_input_node == 0) {
      param_.num_input_node = static_cast<int>(nodes_in[0]->data.shape[0]);
    } else {
      utils::Check(param_.num_input_node == static_cast<int>(nodes_in[0]->data.shape[0]),
                   "FullcLayer: input hidden nodes is not consistent");
    }
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    this->Forward_(is_train, wmat_, nodes_in[0], nodes_out[0]);
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    this->Backprop_(prop_grad, wmat_, nodes_in[0], nodes_out[0]);
  }

 protected:
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

  /*! \brief random number generator */
  mshadow::Random<xpu> *prnd_;
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
  DropConnLayer(mshadow::Random<xpu> *p_rnd) : Parent(p_rnd) {
    dropout_threshold = 0.0f;
  }
  virtual void SetParam(const char *name, const char* val) {
    Parent::SetParam(name, val);
    if (!strcmp("threshold", name)) dropout_threshold = static_cast<real_t>(atof(val));
  }  
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    Parent::InitConnection(nodes_in, nodes_out, p_cstate);
    p_cstate->states.resize(2);
    mshadow::Shape<4> wshape =
        mshadow::Shape4(1, 1, this->param_.num_hidden, this->param_.num_input_node);
    p_cstate->states[0].Resize(wshape);
    p_cstate->states[1].Resize(wshape);
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> mask = p_cstate->states[0][0][0];
    mshadow::Tensor<xpu, 2> tmpw = p_cstate->states[1][0][0];
    if (is_train) {
      const real_t pkeep = 1.0f - dropout_threshold;
      mask = F<op::threshold>(this->prnd_->uniform(mask.shape), pkeep) * (1.0f / pkeep);
      tmpw = this->wmat_ * mask;
    } else {
      mshadow::Copy(tmpw, this->wmat_);
    }
    Parent::Forward_(is_train, tmpw, nodes_in[0], nodes_out[0]);
  }

  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 2> mask = p_cstate->states[0][0][0];
    mshadow::Tensor<xpu, 2> tmpw = p_cstate->states[1][0][0];    
    Parent::Backprop_(prop_grad, tmpw, nodes_in[0], nodes_out[0]);
    Parent::gwmat_ *= mask;
  }  

 private:
  mshadow::real_t dropout_threshold;
  typedef FullConnectLayer<xpu> Parent;
};  // class DropconnLayer
}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_FULLC_LAYER_INL_HPP_
