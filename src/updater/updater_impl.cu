#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "updater_impl-inl.hpp"
// specialize the gpu implementation
namespace cxxnet {
namespace updater {
template<>
void CreateUpdaters<gpu>(const char *type,
                         mshadow::Random<gpu> *p_rnd,
                         layer::ILayer<gpu> *p_layer,
                         std::vector<IUpdater<gpu>*> *out_updaters) {
  CreateUpdaterVisitor<gpu> visitor(type, p_rnd, out_updaters);
  p_layer->ApplyVisitor(&visitor);
}
template<>
void CreateAsyncUpdaters<gpu>(int data_key_base,
                              int device_id,
                              mshadow::ps::IParamServer<gpu, real_t> *param_server,
                              const char *type,
                              mshadow::Random<gpu> *p_rnd,
                              layer::LayerType layer_type,
                              layer::ILayer<gpu> *p_layer,
                              std::vector<IAsyncUpdater<gpu>*> *out_updaters) {
  CreateAsyncUpdaterVisitor<gpu> visitor(data_key_base, device_id, param_server,
                                         type, p_rnd, layer_type, out_updaters);  
  p_layer->ApplyVisitor(&visitor);  
}
}  // namespace updater
}  // namespace cxxnet
