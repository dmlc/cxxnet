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
#include "./fixconn_layer-inl.hpp"
#include "./lrn_layer-inl.hpp"
#include "./flatten_layer-inl.hpp"
#include "./pooling_layer-inl.hpp"
#include "./pairtest_layer-inl.hpp"
#include "./concat_layer-inl.hpp"
#include "./cudnn_convolution_layer-inl.hpp"
#include "./split_layer-inl.hpp"
#include "./cudnn_pooling_layer-inl.hpp"
#include "./xelu_layer-inl.hpp"
#include "./insanity_layer-inl.hpp"
#include "./insanity_pooling_layer-inl.hpp"
#include "./prelu_layer-inl.hpp"
#include "./batch_norm_layer-inl.hpp"
#include "./loss/softmax_layer-inl.hpp"
#include "./loss/lp_loss_layer-inl.hpp"
#include "./loss/multi_logistic_layer-inl.hpp"
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
    case kConv: return new CuDNNConvolutionLayer<xpu>(p_rnd);
    case kBias: return new BiasLayer<xpu>();
    case kDropout: return new DropoutLayer<xpu>(p_rnd);
    case kFullConnect: return new FullConnectLayer<xpu>(p_rnd);
    case kFixConnect: return new FixConnectLayer<xpu>();
    case kLRN: return new LRNLayer<xpu>();
    case kFlatten: return new FlattenLayer<xpu>();
    case kReluMaxPooling: return
        new PoolingLayer<mshadow::red::maximum, false, xpu, false, op::relu, op::relu_grad>();
    case kMaxPooling: return new CuDNNPoolingLayer<mshadow::red::maximum, kMaxPooling, xpu>();
    case kSumPooling: return new PoolingLayer<mshadow::red::sum, kSumPooling, xpu>();
    case kAvgPooling: return new CuDNNPoolingLayer<mshadow::red::sum, kAvgPooling, xpu>();
    case kSoftmax: return new SoftmaxLayer<xpu>(label_info);
    case kConcat: return new ConcatLayer<xpu, 3>();
    case kChConcat: return new ConcatLayer<xpu, 1>();
    case kSplit: return new SplitLayer<xpu>();
    case kXelu: return new XeluLayer<xpu>();
    case kInsanity: return new InsanityLayer<xpu>(p_rnd);
    case kInsanityPooling: return new InsanityPoolingLayer<mshadow::red::maximum, kMaxPooling, xpu>(p_rnd);
    case kPRelu: return new PReluLayer<xpu>(p_rnd);
    case kBatchNorm: return new BatchNormLayer<xpu, true>(p_rnd);
    case kBatchNorm_no_ma: return new BatchNormLayer<xpu, false>(p_rnd);
    case kLpLoss: return new LpLossLayer<xpu>(label_info);
    case kMultiLogistic: return new MultiLogisticLayer<xpu>(label_info);
    #if CXXNET_USE_CAFFE_ADAPTOR
    case kCaffe: return new CaffeLayer<xpu>();
    #endif
    default: utils::Error("unknown layer type id : \"%d\"", type); return NULL;
  }
}

}  // namespace layer
}  // namespace cxxnet
#endif
