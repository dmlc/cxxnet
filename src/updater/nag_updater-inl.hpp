#ifndef CXXNET_UPDATER_NAG_UPDATER_INL_HPP_
#define CXXNET_UPDATER_NAG_UPDATER_INL_HPP_
/*!
 * \file nag_updater-inl.hpp
 * \brief implementation of NAG with momentum
 * \author Winsty
 */
#include <mshadow/tensor.h>
 #include "./updater.h"
#include "./param.h"

namespace cxxnet {
namespace updater {
// SGD updater with momentum
template<typename xpu, int dim>
class NAGUpdater : public IUpdater<xpu> {
 public:
  NAGUpdater(mshadow::Tensor<xpu,dim> w, mshadow::Tensor<xpu,dim> dw, const char *tag)
      :w(w), dw(dw) {
    param.tag = tag;
  }
  virtual ~NAGUpdater(void) {}
  virtual void Init(void) {
    if (param.silent == 0) {
      printf("NAGUpdater: eta=%f, mom=%f\n", param.base_lr_, param.momentum);
    }
    m_w.Resize(w.shape_, 0.0f);
    old_m_w.Resize(w.shape_, 0.0f);
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    w.set_stream(stream);
    dw.set_stream(stream);
    m_w.set_stream(stream);
    old_m_w.set_stream(stream);
  }
  virtual void Update(long epoch) {
    param.ScheduleEpoch(epoch);
    mshadow::Copy(old_m_w, m_w, old_m_w.stream_);
    m_w *= param.momentum;
    m_w += (-param.learning_rate) * (dw + param.wd * w);
    w += (1 + param.momentum) * m_w - param.momentum * old_m_w;
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
  mshadow::TensorContainer<xpu,dim> m_w, old_m_w;
};  // class SGDUpdater
}  // namespace updater
}  // namespace cxxnet
#endif

