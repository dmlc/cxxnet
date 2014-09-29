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
                              const std::vector< Node<gpu> *> &nodes_in,
                              const std::vector< Node<gpu> *> &nodes_out);
}  // namespace layer
}  // namespace cxxnet

