#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// include the layer, this is where the actual implementations are
#include "layer_impl-inl.hpp"
// specialize the gpu implementation here
namespace cxxnet {
namespace layer {
template<>
ILayer<gpu>* CreateLayer<gpu>(LayerType type,
                              mshadow::Random<gpu> *p_rnd,
                              const LabelInfo *label_info) {
  return CreateLayer_<gpu>(type, p_rnd, label_info); 
}
}  // namespace layer
}  // namespace cxxnet

