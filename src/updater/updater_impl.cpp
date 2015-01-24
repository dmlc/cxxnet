#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// this is where the actual implementations are
#include "updater_impl-inl.hpp"
// specialize the cpu implementation
namespace cxxnet {
namespace updater {
template<>
IUpdater<cpu>* CreateUpdater<>(const char *type,
                               mshadow::Random<cpu> *p_rnd,
                               mshadow::Tensor<cpu, 2> weight,
                               mshadow::Tensor<cpu, 2> wgrad,
                               const char *tag) {
  return CreateUpdater_(type, p_rnd, weight, wgrad, tag);
}
template<>
void CreateAsyncUpdaters<cpu>(int layer_index,
                              int device_id,
                              mshadow::ps::ISharedModel<cpu, real_t> *param_server,
                              const char *type,
                              mshadow::Random<cpu> *p_rnd,
                              layer::LayerType layer_type,
                              layer::ILayer<cpu> *p_layer,
                              std::vector<IAsyncUpdater<cpu>*> *out_updaters) {
  CreateAsyncUpdaterVisitor<cpu> visitor(layer_index, device_id, param_server,
                                         type, p_rnd, layer_type, out_updaters);  
  p_layer->ApplyVisitor(&visitor);  
}
}  // namespace updater
}  // namespace cxxnet
