#ifndef CXXNET_UPDATER_SGD_UPDATER_INL_HPP_
#define CXXNET_UPDATER_SGD_UPDATER_INL_HPP_
/*!
 * \file sgd_updater-inl.hpp
 * \brief implementation of SGD with momentum
 * \author Tianqi Chen
 */
#include <mshadow/tensor.h>
#include <cmath>
#include "./updater.h"
#include "./param.h"

namespace cxxnet {
namespace updater {
/*! \brief used for gradient clipping and nan detection */
struct clip {
  MSHADOW_XINLINE static real_t Map(real_t a, real_t b) {
    using namespace std;
    if (isnan(a)) return 0.0f;
    if (a < -b) return -b;
    if (a > b) return b;
    return a;
  }
};

// SGD updater with momentum
template<typename xpu, int dim>
class SGDUpdater : public IUpdater<xpu> {
 public:
  SGDUpdater(mshadow::Tensor<xpu,dim> w, mshadow::Tensor<xpu,dim> dw, const char *tag)
      :w(w), dw(dw) {
    param.tag = tag;
  }
  virtual ~SGDUpdater(void) {}
  virtual void Init(void) {
    if (param.silent == 0) {
      printf("SGDUpdater: eta=%f, mom=%f\n", param.base_lr_, param.momentum);
    }
    m_w.Resize(w.shape_, 0.0f);
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    w.set_stream(stream);
    dw.set_stream(stream);
    m_w.set_stream(stream);
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
  // update function
  virtual void ApplyUpdate(long epoch,
                           mshadow::Tensor<xpu, dim> grad) {
    using namespace mshadow::expr;
    param.ScheduleEpoch(epoch);
    m_w *= param.momentum;
    if (param.clip_gradient != 0.0f) {
      m_w += (-param.learning_rate) * (F<clip>(grad, param.clip_gradient) + param.wd * w);
    } else {
      m_w += (-param.learning_rate) * (grad + param.wd * w);
    }
    w += m_w;
  }
};  // class SGDUpdater
}  // namespace updater
}  // namespace cxxnet
#endif

