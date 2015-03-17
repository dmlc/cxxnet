#ifndef CXXNET_UPDATER_ADAM_UPDATER_INL_HPP_
#define CXXNET_UPDATER_ADAM_UPDATER_INL_HPP_
/*!
 * \file sgd_updater-inl.hpp
 * \brief implementation of SGD with momentum
 * \author Bing Xu
 */
#include <mshadow/tensor.h>
#include "./updater.h"
#include "./param.h"
#include "../layer/op.h"

namespace cxxnet {
namespace updater {
// Adam updater with momentum
template<typename xpu, int dim>
class AdamUpdater : public IUpdater<xpu> {
 public:
  AdamUpdater(mshadow::Tensor<xpu,dim> w, mshadow::Tensor<xpu,dim> dw, const char *tag)
      :w(w), dw(dw) {
    param.tag = tag;
    decay1 = 0.1f;
    decay2 = 0.001f;
  }
  virtual ~AdamUpdater(void) {}
  virtual void Init(void) {
    if (param.silent == 0) {
      printf("AdamUpdater: eta=%f, beta1=%f, beta2=%f\n", param.base_lr_, decay1, decay2);
    }
    m_w1.Resize(w.shape_, 0.0f);
    m_w2.Resize(w.shape_, 0.0f);
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    w.set_stream(stream);
    dw.set_stream(stream);
    m_w1.set_stream(stream);
    m_w2.set_stream(stream);
  }
  virtual void Update(long epoch) {
    this->ApplyUpdate(epoch, dw);
    // dw accumulate gradient instead of storing them
    // updater need to reset then to 0 after each update
    dw = 0.0f;
  }
  virtual void Update(long epoch, mshadow::Tensor<xpu, 2> grad) {
    utils::Assert(grad.shape_ == w.shape_.FlatTo2D(),
                  "SGDUpdater: grad must be generated from source of same shape");
    this->ApplyUpdate(epoch, mshadow::Tensor<xpu, dim>
                      (grad.dptr_, w.shape_, grad.stride_, w.stream_));
  }
  virtual void StartRound(int round) {
    param.round = round;
  }
  virtual void SetParam(const char *name, const char *val) {
    param.SetParam(name, val);
    if (!strcmp(name, "beta1")) decay1 = atof(val);
    if (!strcmp(name, "beta2")) decay2 = atof(val);
  }
  virtual void ApplyVisitor(typename IUpdater<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit(param.tag.c_str(), w, dw);
  }

 protected:
  UpdaterParam param;
  // variales
  mshadow::Tensor<xpu,dim> w, dw;
  // momentum variable
  mshadow::TensorContainer<xpu,dim> m_w1;
  mshadow::TensorContainer<xpu,dim> m_w2;
  float decay1;
  float decay2;
  // update function
  virtual void ApplyUpdate(long epoch,
                           mshadow::Tensor<xpu, dim> grad) {
    if (param.wd > 0.0f) grad -= param.wd * w;
    float fix1 = 1.0f - powf(1.0f - decay1, epoch + 1);
    float fix2 = 1.0f - powf(1.0f - decay2, epoch + 1);
    float lr_t = param.base_lr_ * sqrt(fix2) / fix1;
    m_w1 += decay1 * (grad - m_w1);
    m_w2 += decay2 * (mshadow::expr::F<op::square>(grad) - m_w2);
    w -= lr_t * (m_w1 / (mshadow::expr::F<op::square_root>(m_w2) + 1e-8f));
  }
};  // class AdamUpdater
}  // namespace updater
}  // namespace cxxnet
#endif

