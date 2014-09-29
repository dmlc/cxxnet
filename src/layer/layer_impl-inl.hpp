#ifndef CXXNET_LAYER_IMPL_INL_HPP_
#define CXXNET_LAYER_IMPL_INL_HPP_
/*!
 * \file layer-inl.hpp
 * \brief this file compiles all implementation of layers together
 * \author Bing Xu, Tianqi Chen
 */
#include "./layer.h"
#include "fullc_layer-inl.hpp"
namespace cxxnet {
namespace layer {
template<typename xpu>
ILayer<xpu>* CreateLayer_(LayerType type,
                          mshadow::Random<xpu> *p_rnd,
                          const std::vector< Node<xpu> *> &nodes_in,
                          const std::vector< Node<xpu> *> &nodes_out){
  utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
               "this layer can only take one input and output ");
  switch(type) {
    case kFullConnect: return new FullConnectLayer<xpu>(p_rnd, nodes_in[0], nodes_out[0]);
    default: utils::Error("unknown layer type");
  }
  return NULL;
}

}  // namespace layer
}  // namespace cxxnet
#endif
