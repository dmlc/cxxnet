#ifndef CXXNET_UPDATER_UPDATER_IMPL_INL_HPP_
#define CXXNET_UPDATER_UPDATER_IMPL_INL_HPP_
/*!
 * \file updater_impl-inl.hpp
 * \brief this file compiles all implementations of updaters together
 * \author Tianqi Chen
 */
#include "./sgd_updater-inl.hpp"
#include "./ext_updater-inl.hpp"
#include "./async_updater-inl.hpp"
namespace cxxnet {
namespace updater {
/*!
 * \brief factory: create an upadater algorithm of given type
 */
template<typename xpu, int dim>
inline IUpdater<xpu>* CreateUpdater_(const char *type,
                                     mshadow::Random<xpu> *p_rnd,
                                     mshadow::Tensor<xpu,dim> weight,
                                     mshadow::Tensor<xpu,dim> wgrad,
                                     const char *tag) {
  if(!strcmp(type, "sgd")) return new SGDUpdater<xpu,dim>(weight, wgrad, tag);
  if(!strcmp(type, "sghmc")) return new SGHMCUpdater<xpu,dim>(p_rnd, weight, wgrad, tag);
  if(!strcmp(type, "noise_sgd")) return new NoiseSGDUpdater<xpu, dim>(weight, wgrad, tag, p_rnd);
  utils::Error("unknown updater type %s", type);
  return NULL;
}

template<typename xpu, int dim>
inline IAsyncUpdater<xpu>*
CreateAsyncUpdater_(int data_key,
                    int devid,
                    int priority,
                    mshadow::ps::IParamServer<xpu, real_t> *pserver,
                    const char *type,
                    mshadow::Random<xpu> *p_rnd,
                    layer::LayerType layer_type,
                    mshadow::Tensor<xpu,dim> weight,
                    mshadow::Tensor<xpu,dim> wgrad,
                    const char *tag) {
  return new AsyncUpdater<xpu>(data_key, devid, priority,
                               weight.FlatTo2D(), wgrad.FlatTo2D(), layer_type, tag,
                               pserver, CreateUpdater_(type, p_rnd, weight, wgrad, tag));
}

template<typename xpu>
struct CreateUpdaterVisitor : public IUpdater<xpu>::IVisitor {
  // type of updater
  const char *type;
  // random number generator
  mshadow::Random<xpu> *p_rnd;
  // output updaters
  std::vector<IUpdater<xpu>*> *out_updaters;
  // constructor
  CreateUpdaterVisitor(const char *type,
                       mshadow::Random<xpu> *p_rnd,
                       std::vector<IUpdater<xpu>*> *out_updaters)
      : type(type), p_rnd(p_rnd), out_updaters(out_updaters) {}
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,1> weight,
                     mshadow::Tensor<xpu,1> grad) {
    out_updaters->push_back(CreateUpdater_(type, p_rnd, weight, grad, field_name));
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,2> weight,
                     mshadow::Tensor<xpu,2> grad) {
    out_updaters->push_back(CreateUpdater_(type, p_rnd, weight, grad, field_name));
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,3> weight,
                     mshadow::Tensor<xpu,3> grad) {
    out_updaters->push_back(CreateUpdater_(type, p_rnd, weight, grad, field_name));
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,4> weight,
                     mshadow::Tensor<xpu,4> grad) {
    out_updaters->push_back(CreateUpdater_(type, p_rnd, weight, grad, field_name));
  }
};


template<typename xpu>
struct CreateAsyncUpdaterVisitor : public IUpdater<xpu>::IVisitor {
  // data_key
  int data_key;
  // device id
  int devid;
  // parameter server
  mshadow::ps::IParamServer<xpu, real_t> *pserver;
  // type of updater
  const char *type;
  // random number generator
  mshadow::Random<xpu> *p_rnd;
  // layer type;
  layer::LayerType layer_type;
  // output updaters
  std::vector<IAsyncUpdater<xpu>*> *out_updaters;
  // constructor
  CreateAsyncUpdaterVisitor
  (int data_key,
   int devid,
   mshadow::ps::IParamServer<xpu, real_t> *pserver,
   const char *type,
   mshadow::Random<xpu> *p_rnd,
   layer::LayerType layer_type,
   std::vector<IAsyncUpdater<xpu>*> *out_updaters)
      : data_key(data_key),
        devid(devid),
        pserver(pserver),
        type(type), p_rnd(p_rnd),
        layer_type(layer_type),
        out_updaters(out_updaters) {}
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,1> weight,
                     mshadow::Tensor<xpu,1> grad) {
    out_updaters->push_back(CreateAsyncUpdater_(data_key, devid, -data_key, pserver,
                                                type, p_rnd, layer_type,
                                                weight, grad, field_name));
    data_key += 1;
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,2> weight,
                     mshadow::Tensor<xpu,2> grad) {
    out_updaters->push_back(CreateAsyncUpdater_(data_key, devid, -data_key, pserver,
                                                type, p_rnd, layer_type,
                                                weight, grad, field_name));
    data_key += 1;
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,3> weight,
                     mshadow::Tensor<xpu,3> grad) {
    out_updaters->push_back(CreateAsyncUpdater_(data_key, devid, -data_key, pserver,
                                                type, p_rnd, layer_type,
                                                weight, grad, field_name));
    data_key += 1;
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,4> weight,
                     mshadow::Tensor<xpu,4> grad) {
    out_updaters->push_back(CreateAsyncUpdater_(data_key, devid, -data_key, pserver,
                                                type, p_rnd, layer_type,
                                                weight, grad, field_name));
    data_key += 1;
  }

};

}  // namespace updater
}  // namespace cxxnet
#endif // CXXNET_UPDATER_INL_HPP
