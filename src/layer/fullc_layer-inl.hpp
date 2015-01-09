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
    param_.RandInitWeight(this->prnd_, wmat_, wmat_.size(1), wmat_.size(0));
    bias_ = param_.init_bias;
    // setup gradient weight
    gwmat_.Resize(wmat_.shape_);
    gbias_.Resize(bias_.shape_);
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
    gwmat_.Resize(wmat_.shape_);
    gbias_.Resize(bias_.shape_);
    gwmat_ = 0.0f; gbias_ = 0.0f;
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    // stream of wmat and bias may be reset, but it is ok
    wmat_.set_stream(stream);
    bias_.set_stream(stream);
    gwmat_.set_stream(stream);
    gbias_.set_stream(stream);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "DropoutLayer Layer only support 1-1 connection");
    utils::Check(nodes_in[0]->is_mat(), "input need to be a matrix");
    utils::Check(param_.num_hidden > 0, "must set nhidden correctly");
    // we change matrix convention
    nodes_out[0]->data.shape_ =
        mshadow::Shape4(nodes_in[0]->data.size(0), 1, 1, param_.num_hidden);
    if (param_.num_input_node == 0) {
      param_.num_input_node = static_cast<int>(nodes_in[0]->data.size(3));
    } else {
      utils::Check(param_.num_input_node == static_cast<int>(nodes_in[0]->data.size(3)),
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
    index_t nbatch = m_in.size(0);
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

}  // namespace layer
}  // namespace cxxnet
#endif  // LAYER_FULLC_LAYER_INL_HPP_
