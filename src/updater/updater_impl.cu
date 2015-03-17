#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "updater_impl-inl.hpp"
// specialize the gpu implementation
namespace cxxnet {
namespace updater {
template<>
IUpdater<gpu>* CreateUpdater<gpu>(const char *type,
                                  mshadow::Random<gpu> *p_rnd,
                                  mshadow::Tensor<gpu, 2> weight,
                                  mshadow::Tensor<gpu, 2> wgrad,
                                  const char *tag) {
  return CreateUpdater_(type, p_rnd, weight, wgrad, tag);
}
template<>
void CreateAsyncUpdaters<gpu>(int layer_index,
                              int device_id,
                              mshadow::ps::ISharedModel<gpu, real_t> *param_server,
                              const char *type,
                              mshadow::Random<gpu> *p_rnd,
                              layer::LayerType layer_type,
                              layer::ILayer<gpu> *p_layer,
                              std::vector<IAsyncUpdater<gpu>*> *out_updaters) {
  CreateAsyncUpdaterVisitor<gpu> visitor(layer_index, device_id, param_server,
                                         type, p_rnd, layer_type, out_updaters);
  p_layer->ApplyVisitor(&visitor);
}
}  // namespace updater
}  // namespace cxxnet
