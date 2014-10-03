#ifndef CXXNET_UPDATER_SGD_UPDATER_INL_HPP_
#define CXXNET_UPDATER_SGD_UPDATER_INL_HPP_
/*!
 * \file sgd_updater-inl.hpp
 * \brief implementation of SGD with momentum
 * \author Tianqi Chen
 */
#include <mshadow/tensor.h>
#include "./updater.h"
#include "./param.h"

namespace cxxnet {
namespace updater {
// SGD updater with momentum
template<typename xpu, int dim>
class SGDUpdater : public IUpdater<xpu> {
 public:
  SGDUpdater(mshadow::Tensor<xpu,dim> w, mshadow::Tensor<xpu,dim> dw, const char *tag)
      :w(w), dw(dw) {
    param.tag = tag;
    m_w.Resize(w.shape, 0.0f);
  }
  virtual ~SGDUpdater(void) {}
  virtual void Init(void) {
    if (param.silent == 0) {
      printf("SGDUpdater: eta=%f, mom=%f\n", param.base_lr_, param.momentum);
    }
  }
  virtual void Update(long epoch) {
    param.ScheduleEpoch(epoch);
    m_w *= param.momentum;
    m_w += (-param.learning_rate) * (dw + param.wd * w);
    w += m_w;
    // dw accumulate gradient instead of storing them, updater need to reset then to 0 after each update
    dw = 0.0f;
  }
  virtual void StartRound(int round) {
    param.round = round;
  }
  virtual void SetParam(const char *name, const char *val) {
    param.SetParam(name, val);
  }
  virtual void ApplyVisitor(typename IUpdater<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit(param.tag.c_str(), w, dw);
  }

 protected:
  UpdaterParam param;
  // variales
  mshadow::Tensor<xpu,dim> w, dw;
  // momentum variable
  mshadow::TensorContainer<xpu,dim> m_w;
};  // class SGDUpdater
}  // namespace updater
}  // namespace cxxnet
#endif

