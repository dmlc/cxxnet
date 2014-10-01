#ifndef CXXNET_LAYER_IMPL_INL_HPP_
#define CXXNET_LAYER_IMPL_INL_HPP_
/*!
 * \file layer-inl.hpp
 * \brief this file compiles all implementation of layers together
 * \author Bing Xu, Tianqi Chen
 */
#include "./layer.h"
#include "./fullc_layer-inl.hpp"
#include "./bias_layer-inl.hpp"
#include "./convolution_layer-inl.hpp"
#include "./activation_layer-inl.hpp"
#include "./dropout_layer-inl.hpp"
#include "./lrn_layer-inl.hpp"
#include "./flatten_layer-inl.hpp"
#include "./pooling_layer-inl.hpp"
#include "./softmax_layer-inl.hpp"

namespace cxxnet {
namespace layer {
template<typename xpu>
ILayer<xpu>* CreateLayer_(LayerType type,
                          mshadow::Random<xpu> *p_rnd,
                          const std::vector< Node<xpu> *> &nodes_in,
                          const std::vector< Node<xpu> *> &nodes_out) {
  // code for handling multiple connections return before here
  utils::Check(nodes_in.size() == 1 && nodes_out.size() == 1,
               "this layer can only take one input and output ");
  Node<xpu> *p_in = nodes_in[0];
  Node<xpu> *p_out = nodes_in[0];
  switch(type) {
    case kFullConnect: return new FullConnectLayer<xpu>(p_rnd, p_in, p_out);
    case kBias: return new BiasLayer<xpu>(p_rnd, p_in, p_out);
    case kConv: return new ConvolutionLayer<xpu>(p_rnd, p_in, p_out);
    case kDropout: return new DropoutLayer<xpu>(p_rnd, p_in, p_out);
    case kFlatten: return new FlattenLayer<xpu>(p_rnd, p_in, p_out);
    case kSigmoid: return new ActivationLayer<xpu, op::sigmoid, op::sigmoid_grad>(p_rnd, p_in, p_out);
    case kTanh: return new ActivationLayer<xpu, op::tanh, op::tanh_grad>(p_rnd, p_in, p_out);
    case kRectifiedLinear: return new ActivationLayer<xpu, op::relu, op::relu_grad>(p_rnd, p_in, p_out);
    case kSoftplus: return new ActivationLayer<xpu, op::softplus, op::softplus_grad>(p_rnd, p_in, p_out);
    case kLRN: return new LRNLayer<xpu>(p_rnd, p_in, p_out);
    case kMaxPooling: return new PoolingLayer<mshadow::red::maximum, false, xpu>(p_rnd, p_in, p_out);
    case kSumPooling: return new PoolingLayer<mshadow::red::sum, false, xpu>(p_rnd, p_in, p_out);
    case kAvgPooling: return new PoolingLayer<mshadow::red::sum, true, xpu>(p_rnd, p_in, p_out);
    case kSoftmax: return new SoftmaxLayer<xpu>(p_rnd, p_in, p_out);
    default: utils::Error("unknown layer type");
  }
  return NULL;
}

}  // namespace layer
}  // namespace cxxnet
#endif
