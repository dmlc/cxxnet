#ifndef CXXNET_SYNC_SYNC_IMPL_INL_HPP_
#define CXXNET_SYNC_SYNC_IMPL_INL_HPP_
/*!
 * \file sync_impl-inl.hpp
 * \brief this file includes the implementation of synchronizer
 * \author Tianqi Chen
 */
#include <vector>
#include <mshadow/tensor.h>
#include "./sync.h"

namespace cxxnet {
namespace sync {
template<typename xpu>
class SimpleSynch : public ISynchronizer<xpu> {
 public:
  SimpleSynch(const char *tag,
              const std::vector< mshadow::Tensor<xpu,2> > &weights,
              const std::vector< mshadow::Tensor<xpu,2> > &grads,
              const std::vector<int> devices)
      : tag(tag), weights(weights), grads(grads), devices(devices) {
    utils::Assert(weights.size() != 0, "empty list");
    utils::Assert(weights.size() == grads.size(), "SimpleSynch grad weights size mismatch");
    for (size_t i = 1; i < weights.size(); ++i) {
      utils::Assert(weights[i].shape == weights[0].shape, "SimpleSynch shape mismatch");
      utils::Assert(weights[i].shape.MSize() == weights[0].shape.MSize(), "SimpleSynch shape mismatch");
    }            
    for (size_t i = 0; i < grads.size(); ++i) {
      utils::Assert(grads[i].shape == weights[0].shape, "SimpleSynch shape mismatch");
      utils::Assert(grads[i].shape.MSize() == weights[0].shape.MSize(), "SimpleSynch shape mismatch");
    }
    host_device = 0;
#ifdef __CUDACC__
    if (!xpu::kDevCPU){
      utils::Check(cudaGetDevice(&host_device) == cudaSuccess, "cannot get device");
    }
#endif
    // no synchronization is needed for 1 weight
    if (weights.size() > 1) {
      wtmp.Resize(mshadow::Shape1(weights[0].shape.MSize()));
      wsum.Resize(mshadow::Shape1(weights[0].shape.MSize()));
    }
    // default parameters
    // by default we also synchronize weight
    // setting it to 0 means we only synchronize gradient
    sync_weight = 1;
  }
  virtual ~SimpleSynch(void){}
  virtual void SetParam(const char *name, const char *val) {
    if (!strncmp(name, tag.c_str(), tag.length())) {
      if (name[tag.length()] == ':') name += tag.length() + 1;
    }
    if (!strcmp(name, "sync_weight")) sync_weight = atoi(val);
  }  
  /*! \brief synchronization actions to be performs before the updater */
  virtual void SyncBeforeUpdate(void) {
    if (weights.size() == 1) return;
    // sync gradient
    wsum = 0.0f;
    for (size_t i = 0; i < grads.size(); ++i) {
      Copy(wtmp.dptr, host_device, 
           grads[i].dptr, devices[i],
           sizeof(mshadow::real_t) * wtmp.shape[0]);
      wsum += wtmp;      
    }
    for (size_t i = 0; i < grads.size(); ++i) {
      Copy(grads[i].dptr, devices[i],
           wsum.dptr, host_device, 
           sizeof(mshadow::real_t) * wsum.shape[0]);
    }
  }
  /*! \brief synchronization actions to be performs before the updater */
  virtual void SyncAfterUpdate(void) {
    if (weights.size() == 1|| sync_weight == 0) return;
    for (size_t i = 1; i < weights.size(); ++i) {
      Copy(weights[i].dptr, devices[i],
           weights[0].dptr, devices[0], 
           sizeof(mshadow::real_t) * wsum.shape[0]);
    }
  }
  inline static void Copy(real_t *dst, int dst_dev,
                          real_t *src, int src_dev,
                          size_t size) {
    #ifdef __CUDACC__
    if (!xpu::kDevCPU){
      cudaMemcpyPeer(dst, dst_dev, src, src_dev, size);
    } else {
      memcpy(dst, src, size);
    }
    #else
    memcpy(dst, src, size);
    #endif    
  }

 private:
  int sync_weight;
  std::string tag;
  int host_device;
  mshadow::TensorContainer<xpu, 1> wtmp, wsum;
  std::vector< mshadow::Tensor<xpu,2> > weights;
  std::vector< mshadow::Tensor<xpu,2> > grads;
  std::vector<int> devices;
};


template<typename xpu>
ISynchronizer<xpu>* CreateSynch(const char *type,
                                const std::vector< mshadow::Tensor<xpu,2> > &weights,
                                const std::vector< mshadow::Tensor<xpu,2> > &grads,
                                const std::vector<int> devices,
                                const char *tag) {
  if (!strcmp(type, "none")) return NULL;
  if (!strcmp(type, "simple")) return new SimpleSynch<xpu>(tag, weights, grads, devices);
  utils::Error("unknown syncrhonizer type %s", type);
  return NULL;
}  
}  // namespace sync
}  // namespace cxxnet
#endif

