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
}  // namespace updater
}  // namespace cxxnet
