#ifndef CXXNET_UPDATER_ASYNC_UPDATER_INL_HPP_
#define CXXNET_UPDATER_ASYNC_UPDATER_INL_HPP_
/*!
 * \file sgd_updater-inl.hpp
 * \brief implementation of SGD with momentum
 * \author Tianqi Chen
 */
#include <mshadow/tensor.h>
#include <mshadow-ps/ps.h>
#include "./updater.h"
namespace cxxnet {
namespace updater {
template<typename xpu>
class AsyncUpdater: public IAsyncUpdater<xpu> {
 public:
  AsyncUpdater(int data_key, int devid, int priority,
               mshadow::Tensor<xpu, 2> w, mshadow::Tensor<xpu, 2> dw,
               mshadow::ps::IParamServer<xpu, real_t> *pserver,
               IUpdater<xpu> *updater)
      : data_key(data_key), devid(devid),
        priority(priority), w(w), dw(dw),
        pserver(pserver), updater(updater) {
  }
  virtual ~AsyncUpdater(void) {
    delete updater;
  }
  virtual void Init(void) {
    updater->Init();
  }
  virtual void SetStream(mshadow::Stream<xpu> *stream) {
    updater->SetStream(stream);
  }
  virtual void Update(long epoch) {
    if (pserver == NULL) {
      updater->Update(epoch); return;
    }
    this->update_epoch = epoch;
    pserver->Push(dw, data_key, devid, priority);
    pserver->PullReq(dw, data_key, devid, priority,
                     ApplyUpdate_,
                     this);    
  }
  virtual void UpdateWait(void) {
    if (pserver == NULL) return;
    pserver->PullWait(data_key, devid);
  }
  virtual void StartRound(int round) {
    updater->StartRound(round);
  }
  virtual void SetParam(const char *name, const char *val) {
    updater->SetParam(name, val);
  }
  virtual void ApplyVisitor(typename IUpdater<xpu>::IVisitor *pvisitor) {
    updater->ApplyVisitor(pvisitor);
  }

 protected:
  inline static void ApplyUpdate_(mshadow::Stream<xpu> *stream, void *arg) {
    AsyncUpdater<xpu> *up = static_cast<AsyncUpdater<xpu>*>(arg);
    up->updater->SetStream(stream);
    up->updater->Update(up->update_epoch);
  }
  int data_key, devid, priority;
  int use_ps;
  long update_epoch;
  mshadow::Tensor<xpu, 2> w, dw;
  mshadow::ps::IParamServer<xpu, real_t> *pserver;
  IUpdater<xpu> *updater;
};
}  // updater
}  // cxxnet
#endif

