#ifndef BATCH_NORM_LAYER_INL_HPP
#define BATCH_NROM_LAYER_INL_HPP
#pragma once

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./op.h"


namespace cxxnet {
namespace layer {

template<typename xpu>
class BatchNormLayer : public ILayer<xpu> {
 public:
  BatchNormLayer(mshadow::Random<xpu> *p_rnd) : prnd_(p_rnd) {
    init_slope_ = 0.25f;
    init_bias_ = 0.0f;
    eps_ = 0.01f;
    reset_period_ = -1.0f;
    period_ = 0.0f;
  }
  virtual void SetParam(const char *name, const char* val) {
    if (!strcmp(name, "init_slope")) init_slope_ = atof(val);
    if (!strcmp(name, "init_bias")) init_bias_ = atof(val);
    if (!strcmp(name, "eps")) eps_ = atof(val);
    if (!strcmp(name, "reset_period")) reset_period_ = atof(val);
  }
  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit("wmat", slope_, gslope_);
    pvisitor->Visit("wmat", bias_, gbias_);
  }
  virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
    utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
                 "BNLayer: only support 1-1 connection");
    in_shape_ = nodes_in[0]->data.shape_;
    if (nodes_in[0]->data.size(1) == 1){
      // This is a fc layer
      channel_ = nodes_in[0]->data.size(3);
    } else {
      // This is a conv layer
      channel_ = nodes_in[0]->data.size(1);
    }
    nodes_out[0]->data.shape_ = nodes_in[0]->data.shape_;
  }
  virtual void InitModel(void) {
    temp_.Resize(in_shape_);
    slope_.Resize(mshadow::Shape1(channel_));
    gslope_.Resize(mshadow::Shape1(channel_));
    exp_.Resize(mshadow::Shape1(channel_));
    var_.Resize(mshadow::Shape1(channel_));
    gexp_.Resize(slope_.shape_);
    gvar_.Resize(slope_.shape_);
    hist_exp_.Resize(slope_.shape_);
    hist_var_.Resize(slope_.shape_);
    wtf_.Resize(slope_.shape_);
    bias_.Resize(slope_.shape_);
    gbias_.Resize(slope_.shape_);
    gslope_ = 0.0f;
    gexp_ = 0.0f;
    gvar_ = 0.0f;
  }
  virtual void SaveModel(utils::IStream &fo) const{
    slope_.SaveBinary(fo);
    bias_.SaveBinary(fo);
    hist_exp_.SaveBinary(fo);
    hist_var_.SaveBinary(fo);
  }
  virtual void LoadModel(utils::IStream &fi){
    slope_.LoadBinary(fi);
    bias_.LoadBinary(fi);
    hist_exp_.LoadBinary(fi);
    hist_var_.LoadBinary(fi);
    temp_.Resize(in_shape_);
    gslope_.Resize(slope_.shape_);
    exp_.Resize(slope_.shape_);
    gexp_.Resize(slope_.shape_);
    var_.Resize(slope_.shape_);
    gvar_.Resize(slope_.shape_);
    wtf_.Resize(slope_.shape_);
    gbias_.Resize(slope_.shape_);
    gslope_ = 0.0f;
    gbias_ = 0.0f;
    gexp_ = 0.0f;
    gvar_ = 0.0f;
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    slope_.set_stream(stream);
    gslope_.set_stream(stream);
    exp_.set_stream(stream);
    gexp_.set_stream(stream);
    var_.set_stream(stream);
    gvar_.set_stream(stream);
    temp_.set_stream(stream);
    wtf_.set_stream(stream);
    bias_.set_stream(stream);
    gbias_.set_stream(stream);
    hist_exp_.set_stream(stream);
    hist_var_.set_stream(stream);
  }
  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
                                  const std::vector<Node<xpu>*> &nodes_out,
                                  ConnectState<xpu> *p_cstate) {

  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    utils::Check(reset_period_ > 0.0f, "Reset period must be set for bn layer");
    if (is_train && period_ > reset_period_) {
      period_ = 0.0f;
      hist_exp_ = 0.0f;
      hist_var_ = 0.0f;
    }
    if (is_train) period_ += 1.0f;
    else period_ = reset_period_;
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &out = nodes_out[0]->data;
    const float scale = 1.0f / in.size(0);
    if (is_train) {
      mshadow::Copy(temp_, in, out.stream_);
      if (in.size(1) != 1) {
        exp_ = scale * sumall_except_dim<1>(in);
        var_ = scale * sumall_except_dim<1>(F<op::square>(in - broadcast<1>(exp_, in.shape_)));
        in = (in - broadcast<1>(exp_, in.shape_)) /
          F<op::square_root>(broadcast<1>(var_ + eps_, in_shape_));
        out = in * broadcast<1>(slope_, in.shape_) + broadcast<1>(bias_, in.shape_);
      } else {
        exp_ = scale * sumall_except_dim<3>(in);
        var_ = scale * sumall_except_dim<3>(F<op::square>(in - broadcast<3>(exp_, in.shape_)));
        in = (in - broadcast<3>(exp_, in.shape_)) /
          F<op::square_root>(broadcast<3>(var_ + eps_, in_shape_));
        out = in * broadcast<3>(slope_, in.shape_) + broadcast<3>(bias_, in.shape_);
      }
      hist_exp_ += exp_;
      hist_var_ += var_;
    } else {
      if (in.size(1) != 1) {
        out = broadcast<1>(slope_ / F<op::square_root>(hist_var_ / period_ + eps_), in.shape_) *
          in + broadcast<1>(bias_ - (slope_ * hist_exp_ / period_) /
                            F<op::square_root>(hist_var_ / period_ + eps_), in.shape_);
      } else {
        out = broadcast<3>(slope_ / F<op::square_root>(hist_var_ / period_ + eps_), in.shape_) *
          in + broadcast<3>(bias_ - (slope_ * hist_exp_ / period_) /
                            F<op::square_root>(hist_var_ / period_ + eps_), in.shape_);

      }
    }
  }
  virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &out = nodes_out[0]->data;
    const float scale = 1.0f / in.size(0);
    if (in.size(1) != 1){
      gvar_ = sumall_except_dim<1>((out * broadcast<1>(slope_, in.shape_)) *
                        (temp_ - broadcast<1>(exp_, in.shape_)) *
                        -0.5f * F<op::power>(broadcast<1>(var_ + eps_, in.shape_), -1.5f));
      gexp_ = sumall_except_dim<1>(out * broadcast<1>(slope_, in.shape_));
      gexp_ *= -1.0f / F<op::square_root>(var_ + eps_);
      wtf_ = scale * sumall_except_dim<1>(-2.0f * (temp_ - broadcast<1>(exp_, in.shape_)));
      wtf_ *= gvar_;
      gexp_ += wtf_;
      gslope_ += sumall_except_dim<1>(out * in);
      gbias_ += sumall_except_dim<1>(out);
      in = (out * broadcast<1>(slope_, in.shape_)) *
           broadcast<1>(1.0f / F<op::square_root>(var_ + eps_), in.shape_) +
           broadcast<1>(gvar_, in.shape_) * scale * 2.0f * (temp_ - broadcast<1>(exp_, in.shape_)) +
           broadcast<1>(gexp_, in.shape_) * scale;
    } else {
      gvar_ = sumall_except_dim<3>((out * broadcast<3>(slope_, in.shape_)) *
                        (temp_ - broadcast<3>(exp_, in.shape_)) *
                        -0.5f * F<op::power>(broadcast<3>(var_ + eps_, in.shape_), -1.5f));
      gexp_ = sumall_except_dim<3>(out * broadcast<3>(slope_, in.shape_));
      gexp_ *= -1.0f / F<op::square_root>(var_ + eps_);
      wtf_ = scale * sumall_except_dim<3>(-2.0f * (temp_ - broadcast<3>(exp_, in.shape_)));
      wtf_ *= gvar_;
      gexp_ += wtf_;
      gslope_ += sumall_except_dim<3>(out * in);
      gbias_ += sumall_except_dim<3>(out);
      in = (out * broadcast<3>(slope_, in.shape_)) *
           broadcast<3>(1.0f / F<op::square_root>(var_ + eps_), in.shape_) +
           broadcast<3>(gvar_, in.shape_) * scale * 2.0f * (temp_ - broadcast<3>(exp_, in.shape_)) +
           broadcast<3>(gexp_, in.shape_) * scale;

    }
  }

 private:
  mshadow::Random<xpu> *prnd_;
  int channel_;
  mshadow::Shape<4> in_shape_;
  mshadow::TensorContainer<xpu, 4> temp_;
  mshadow::TensorContainer<xpu, 1> wtf_; // potential mshadow bug
  mshadow::TensorContainer<xpu, 1> slope_;
  mshadow::TensorContainer<xpu, 1> gslope_;
  mshadow::TensorContainer<xpu, 1> bias_;
  mshadow::TensorContainer<xpu, 1> gbias_;
  mshadow::TensorContainer<xpu, 1> exp_;
  mshadow::TensorContainer<xpu, 1> gexp_;
  mshadow::TensorContainer<xpu, 1> var_;
  mshadow::TensorContainer<xpu, 1> gvar_;
  mshadow::TensorContainer<xpu, 1> hist_exp_;
  mshadow::TensorContainer<xpu, 1> hist_var_;
  float init_slope_;
  float init_bias_;
  float eps_;
  float period_;
  float reset_period_;
};  // class BatchNormLayer

} // namespace layer
} // namespace cxxnet
#endif // INSANITY_LAYER_INL_HPP
