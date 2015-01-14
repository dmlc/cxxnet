#ifndef CXXNET_LAYER_IMPL_INL_HPP_
#define CXXNET_LAYER_IMPL_INL_HPP_
/*!
 * \file layer-inl.hpp
 * \brief this file compiles all implementation of layers together
 * \author Bing Xu, Tianqi Chen
 */
#include "./layer.h"
#include "./activation_layer-inl.hpp"
#include "./convolution_layer-inl.hpp"
#include "./bias_layer-inl.hpp"
#include "./dropout_layer-inl.hpp"
#include "./fullc_layer-inl.hpp"
#include "./lrn_layer-inl.hpp"
#include "./flatten_layer-inl.hpp"
#include "./pooling_layer-inl.hpp"
#include "./softmax_layer-inl.hpp"
#include "./pairtest_layer-inl.hpp"
#include "./concat_layer-inl.hpp"
#ifdef __CUDACC__
#include "./cudnn_pooling_layer-inl.hpp"
#include "./cudnn_convolution_layer-inl.hpp"
#endif
#if CXXNET_USE_CAFFE_ADAPTOR
#include "../plugin/caffe_adapter-inl.hpp"
#endif
namespace cxxnet {
namespace layer {
template<typename xpu>
ILayer<xpu>* CreateLayer_(LayerType type,
                          mshadow::Random<xpu> *p_rnd,
                          const LabelInfo *label_info) {
  if (type >= kPairTestGap) {
    return new PairTestLayer<xpu>(CreateLayer_(type / kPairTestGap, p_rnd, label_info),
                                  CreateLayer_(type % kPairTestGap, p_rnd, label_info));
  }
  switch(type) {
    case kSigmoid: return new ActivationLayer<xpu, op::sigmoid, op::sigmoid_grad>();
    case kTanh: return new ActivationLayer<xpu, op::tanh, op::tanh_grad>();
    case kRectifiedLinear: return new ActivationLayer<xpu, op::relu, op::relu_grad>();
    case kConv: return new ConvolutionLayer<xpu>(p_rnd);
    case kBias: return new BiasLayer<xpu>();
    case kDropout: return new DropoutLayer<xpu>(p_rnd);
    case kFullConnect: return new FullConnectLayer<xpu>(p_rnd);
    case kLRN: return new LRNLayer<xpu>();
    case kFlatten: return new FlattenLayer<xpu>();
    case kReluMaxPooling: return
        new PoolingLayer<mshadow::red::maximum, false, xpu, false, op::relu, op::relu_grad>();
    case kMaxPooling: return new PoolingLayer<mshadow::red::maximum, false, xpu>();
    case kSumPooling: return new PoolingLayer<mshadow::red::sum, false, xpu>();
    case kAvgPooling: return new PoolingLayer<mshadow::red::sum, true, xpu>();
    case kSoftmax: return new SoftmaxLayer<xpu>(label_info);
    case kConcat: return new ConcatLayer<xpu>();
#ifdef __CUDACC__
    //case kXelu: return new CUConvolutionLayer<xpu>(p_rnd);
    case kCuDNNConv: return new CuDNNConvolutionLayer<xpu>(p_rnd);
    case kCuDNNMaxPooling: return new CuDNNPoolingLayer<mshadow::red::maximum, false, xpu>();
#endif
    #if CXXNET_USE_CAFFE_ADAPTOR
    case kCaffe: return new CaffeLayer<xpu>();
    #endif
    default: utils::Error("unknown layer type id : \"%d\"", type); return NULL;
  }
}

}  // namespace layer
}  // namespace cxxnet
#endif
