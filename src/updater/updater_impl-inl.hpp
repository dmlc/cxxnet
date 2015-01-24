#ifndef CXXNET_UPDATER_UPDATER_IMPL_INL_HPP_
#define CXXNET_UPDATER_UPDATER_IMPL_INL_HPP_
/*!
 * \file updater_impl-inl.hpp
 * \brief this file compiles all implementations of updaters together
 * \author Tianqi Chen
 */
#include "./sgd_updater-inl.hpp"
#include "./async_updater-inl.hpp"
#include "./nag_updater-inl.hpp"
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
  if(!strcmp(type, "nag")) return new NAGUpdater<xpu, dim>(weight, wgrad, tag);
  utils::Error("unknown updater type %s", type);
  return NULL;
}

template<typename xpu, int dim>
inline IAsyncUpdater<xpu>*
CreateAsyncUpdater_(int layer_index,
                    int devid,
                    int priority,
                    mshadow::ps::ISharedModel<xpu, real_t> *pserver,
                    const char *type,
                    mshadow::Random<xpu> *p_rnd,
                    layer::LayerType layer_type,
                    mshadow::Tensor<xpu,dim> weight,
                    mshadow::Tensor<xpu,dim> wgrad,
                    const char *tag) {  
  return new AsyncUpdater<xpu>(EncodeDataKey(layer_index, tag),
                               devid, priority,
                               weight.FlatTo2D(), wgrad.FlatTo2D(), layer_type, tag,
                               pserver, CreateUpdater_(type, p_rnd, weight, wgrad, tag));
}

template<typename xpu>
struct CreateAsyncUpdaterVisitor : public IUpdater<xpu>::IVisitor {
  // layerid
  int layerid;
  // device id
  int devid;
  // parameter server
  mshadow::ps::ISharedModel<xpu, real_t> *pserver;
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
  (int layerid,
   int devid,
   mshadow::ps::ISharedModel<xpu, real_t> *pserver,
   const char *type,
   mshadow::Random<xpu> *p_rnd,
   layer::LayerType layer_type,
   std::vector<IAsyncUpdater<xpu>*> *out_updaters)
      : layerid(layerid),
        devid(devid),
        pserver(pserver),
        type(type), p_rnd(p_rnd),
        layer_type(layer_type),
        out_updaters(out_updaters) {}
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,1> weight,
                     mshadow::Tensor<xpu,1> grad) {
    out_updaters->push_back(CreateAsyncUpdater_(layerid, devid, -layerid, pserver,
                                                type, p_rnd, layer_type,
                                                weight, grad, field_name));
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,2> weight,
                     mshadow::Tensor<xpu,2> grad) {
    out_updaters->push_back(CreateAsyncUpdater_(layerid, devid, -layerid, pserver,
                                                type, p_rnd, layer_type,
                                                weight, grad, field_name));
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,3> weight,
                     mshadow::Tensor<xpu,3> grad) {
    out_updaters->push_back(CreateAsyncUpdater_(layerid, devid, -layerid, pserver,
                                                type, p_rnd, layer_type,
                                                weight, grad, field_name));
  }
  virtual void Visit(const char *field_name,
                     mshadow::Tensor<xpu,4> weight,
                     mshadow::Tensor<xpu,4> grad) {
    out_updaters->push_back(CreateAsyncUpdater_(layerid, devid, -layerid, pserver,
                                                type, p_rnd, layer_type,
                                                weight, grad, field_name));
  }

};

}  // namespace updater
}  // namespace cxxnet
#endif // CXXNET_UPDATER_INL_HPP
