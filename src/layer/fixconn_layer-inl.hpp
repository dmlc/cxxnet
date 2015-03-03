#ifndef CXXNET_LAYER_FIXCONN_LAYER_INL_HPP_
#define CXXNET_LAYER_FIXCONN_LAYER_INL_HPP_

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "./op.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {
// layer that fix the con weight
template<typename xpu>
class FixConnectLayer : public ILayer<xpu> {
 public:
  FixConnectLayer(void) {
    fname_weight_ = "NULL";
    init = false;
  }
  virtual void SetParam(const char *name, const char* val) {
    param_.SetParam(name, val);
    if (!strcmp(name, "fixconn_weight")) fname_weight_ = val;
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    wmat_.set_stream(stream);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "FixConnLayer: Layer only support 1-1 connection");
    utils::Check(nodes_in[0]->is_mat(), "FixConnLayer: input need to be a matrix");
    utils::Check(param_.num_hidden > 0, "FixConncLayer: must set nhidden correctly");
    // we change matrix convention
    nodes_out[0]->data.shape_ =
        mshadow::Shape4(nodes_in[0]->data.size(0), 1, 1, param_.num_hidden);
    wmat_.Resize(mshadow::Shape2(param_.num_hidden, nodes_in[0]->mat().size(1)));
    utils::Check(fname_weight_ != "NULL", "FixConnLayer: must specify fixconn_weight");
    // mshadow::TensorContainer<cpu, 2> tmp(false);
    tmp.set_pad(false);
    tmp.Resize(wmat_.shape_); tmp = 0.0f;
    FILE *fi = utils::FopenCheck(fname_weight_.c_str(), "r");
    unsigned nrow, ncol, nonzero;
    utils::Check(fscanf(fi, "%u%u%u", &nrow, &ncol, &nonzero) == 3,
                 "FixConnLayer: fixconn_weight invalid sparse matrix format");
    utils::Check(nrow == tmp.size(0) && ncol == tmp.size(1),
                 "FixConnLayer: fixconn_weight shape do not match architecture");
    while (nonzero--) {
      float value;
      unsigned x, y;
      utils::Check(fscanf(fi, "%u%u%f", &x, &y, &value) == 3,
                   "FixConnLayer: fixconn_weight invalid sparse matrix format");
      utils::Check(x < tmp.size(0) && y < tmp.size(1),
                   "FixConnLayer: fixconn_weight index exceed matrix shape");
      tmp[x][y] = value;
    }
    fclose(fi);
    //mshadow::Stream<xpu> *stream = wmat_.stream_;
    // mshadow::Copy(wmat_, tmp, stream);
    // must wait till copy end, otherwise tmp will be de-allocated
    // if (stream != NULL) stream->Wait();
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    if (!init) {
      mshadow::Copy(wmat_, tmp, wmat_.stream_);
      init = true;
    }
    nodes_out[0]->mat() = dot(nodes_in[0]->mat(), wmat_.T());
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    if (prop_grad) {
      nodes_in[0]->mat() = dot(nodes_out[0]->mat(), wmat_);
    }
  }
 private:
  /*! \brief name to the weight */
  std::string fname_weight_;
  /*! \brief parameters that potentially be useful */
  LayerParam param_;
  /*! \brief weight matrix */
  mshadow::TensorContainer<xpu, 2> wmat_;
  /*! \brief temp weight */
  mshadow::TensorContainer<cpu, 2> tmp;
  bool init;
};
}  // namespace layer
}  // namespace cxxnet
#endif  // CXXNET_LAYER_FIXCONN_LAYER_INL_HPP_
