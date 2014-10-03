#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "updater_impl-inl.hpp"
// specialize the cpu implementation
namespace cxxnet {
namespace updater {
template<>
void CreateUpdaters<cpu>(const char *type,
                         mshadow::Random<cpu> *p_rnd,
                         layer::ILayer<cpu> *p_layer,
                         std::vector<IUpdater<cpu>*> *out_updaters) {
  CreateUpdaterVisitor<cpu> visitor(type, p_rnd, out_updaters);
  p_layer->ApplyVisitor(&visitor);
}
}  // namespace updater
}  // namespace cxxnet
