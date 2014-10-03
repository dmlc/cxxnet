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
                              const std::vector<Node<cpu>*> &nodes_in,
                              const std::vector<Node<cpu>*> &nodes_out) {
  return CreateLayer_<cpu>(type, p_rnd, nodes_in, nodes_out); 
}
}  // namespace layer
}  // namespace cxxnet
