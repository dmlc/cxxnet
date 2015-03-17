#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are

#include "layer_impl-inl.hpp"
// specialize the cpu implementation here
namespace cxxnet {
namespace layer {
template<>
ILayer<cpu>* CreateLayer<cpu>(LayerType type,
                              mshadow::Random<cpu> *p_rnd,
                              const LabelInfo *label_info) {
  return CreateLayer_<cpu>(type, p_rnd, label_info); 
}
}  // namespace layer
}  // namespace cxxnet
