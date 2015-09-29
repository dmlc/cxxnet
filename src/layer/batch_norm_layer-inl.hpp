#ifndef BATCH_NORM_LAYER_INL_HPP_
#define BATCH_NORM_LAYER_INL_HPP_
#pragma once

#include <mshadow/tensor.h>
#include "./layer.h"
#include "./op.h"


namespace cxxnet {
namespace layer {

template<typename xpu, bool moving_avg>
class BatchNormLayer : public ILayer<xpu> {
 public:
  BatchNormLayer(mshadow::Random<xpu> *p_rnd) : prnd_(p_rnd) {
    init_slope_ = 1.0f;
    init_bias_ = 0.0f;
    eps_ = 1e-10f;
    bn_momentum_ = 0.9f;
  }
  virtual void SetParam(const char *name, const char* val) {
    if (!strcmp(name, "init_slope")) init_slope_ = atof(val);
    if (!strcmp(name, "init_bias")) init_bias_ = atof(val);
    if (!strcmp(name, "eps")) eps_ = atof(val);
    if (!strcmp(name, "bn_momentum")) bn_momentum_ = atof(val);
  }
  virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit("weight", slope_, gslope_);
    pvisitor->Visit("bias", bias_, gbias_);
    pvisitor->Visit("gamma", slope_, gslope_);
    pvisitor->Visit("beta", bias_, gbias_);
    pvisitor->Visit("moving_mean", running_exp_, running_exp_);
    pvisitor->Visit("moving_var", running_var_, running_var_);
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
    p_cstate->states.resize(1);
    p_cstate->states[0].Resize(nodes_in[0]->data.shape_);
  }
  virtual void InitModel(void) {
    slope_.Resize(mshadow::Shape1(channel_));
    gslope_.Resize(mshadow::Shape1(channel_));
    exp_.Resize(mshadow::Shape1(channel_));
    var_.Resize(mshadow::Shape1(channel_));
    gexp_.Resize(slope_.shape_);
    gvar_.Resize(slope_.shape_);
    wtf_.Resize(slope_.shape_);
    bias_.Resize(slope_.shape_);
    gbias_.Resize(slope_.shape_);
    if (moving_avg) {
      running_exp_.Resize(slope_.shape_);
      running_var_.Resize(slope_.shape_);
      running_exp_ = 0.0f;
      running_var_ = 0.0f;
    }
    gslope_ = 0.0f;
    gbias_ = 0.0f;
    gexp_ = 0.0f;
    gvar_ = 0.0f;
    slope_ = init_slope_;
    bias_ = init_bias_;
  }
  virtual void SaveModel(utils::IStream &fo) const{
    slope_.SaveBinary(fo);
    bias_.SaveBinary(fo);
    if (moving_avg) {
      running_exp_.SaveBinary(fo);
      running_var_.SaveBinary(fo);
    }
  }
  virtual void LoadModel(utils::IStream &fi){
    slope_.LoadBinary(fi);
    bias_.LoadBinary(fi);
    if (moving_avg) {
      running_exp_.LoadBinary(fi);
      running_var_.LoadBinary(fi);
    }
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
    wtf_.set_stream(stream);
    bias_.set_stream(stream);
    gbias_.set_stream(stream);
    if (moving_avg) {
      running_exp_.set_stream(stream);
      running_var_.set_stream(stream);
    }
  }
  virtual void OnBatchSizeChanged(const std::vector<Node<xpu>*> &nodes_in,
                                  const std::vector<Node<xpu>*> &nodes_out,
                                  ConnectState<xpu> *p_cstate) {
    p_cstate->states[0].Resize(nodes_in[0]->data.shape_);
  }
  virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
    using namespace mshadow::expr;
    mshadow::Tensor<xpu, 4> &in = nodes_in[0]->data;
    mshadow::Tensor<xpu, 4> &out = nodes_out[0]->data;
    float scale = 1.0f / in.shape_.Size() * channel_;
    mshadow::TensorContainer<xpu,4> &temp_ = p_cstate->states[0];
    if (is_train) {
      mshadow::Copy(temp_, in, temp_.stream_);
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
      if (moving_avg) {
        running_exp_ = running_exp_ * bn_momentum_ + exp_ * (1 - bn_momentum_);
        running_var_ = running_var_ * bn_momentum_ + var_ * (1 - bn_momentum_);
      }
    } else {
      if (in.size(1) != 1) {
        if (moving_avg) {
          out = broadcast<1>(slope_ / F<op::square_root>(running_var_ + eps_), in.shape_) *
            in + broadcast<1>(bias_ - (slope_ * running_exp_) /
                            F<op::square_root>(running_var_ + eps_), in.shape_);

        } else {
          exp_ = scale * sumall_except_dim<1>(in);
          var_ = scale * sumall_except_dim<1>(F<op::square>(in - broadcast<1>(exp_, in.shape_)));
          out = broadcast<1>(slope_ / F<op::square_root>(var_ + eps_), in.shape_) *
            in + broadcast<1>(bias_ - (slope_ * exp_) /
                            F<op::square_root>(var_ + eps_), in.shape_);
        }
      } else {
        if (moving_avg) {
          out = broadcast<3>(slope_ / F<op::square_root>(running_var_  + eps_), in.shape_) *
            in + broadcast<3>(bias_ - (slope_ * running_exp_) /
                            F<op::square_root>(running_var_ + eps_), in.shape_);
        } else {
          exp_ = scale * sumall_except_dim<3>(in);
          var_ = scale * sumall_except_dim<3>(F<op::square>(in - broadcast<3>(exp_, in.shape_)));
          out = broadcast<3>(slope_ / F<op::square_root>(var_  + eps_), in.shape_) *
            in + broadcast<3>(bias_ - (slope_ * exp_) /
                            F<op::square_root>(var_ + eps_), in.shape_);
        }
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
    float scale = 1.0f / in.shape_.Size() * channel_;
    mshadow::TensorContainer<xpu,4> &temp_ = p_cstate->states[0];
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
  mshadow::TensorContainer<xpu, 1> wtf_;
  mshadow::TensorContainer<xpu, 1> slope_;
  mshadow::TensorContainer<xpu, 1> gslope_;
  mshadow::TensorContainer<xpu, 1> bias_;
  mshadow::TensorContainer<xpu, 1> gbias_;
  mshadow::TensorContainer<xpu, 1> exp_;
  mshadow::TensorContainer<xpu, 1> gexp_;
  mshadow::TensorContainer<xpu, 1> var_;
  mshadow::TensorContainer<xpu, 1> gvar_;
  mshadow::TensorContainer<xpu, 1> running_exp_;
  mshadow::TensorContainer<xpu, 1> running_var_;
  float init_slope_;
  float init_bias_;
  float eps_;
  float bn_momentum_;
};  // class BatchNormLayer

} // namespace layer
} // namespace cxxnet
#endif
