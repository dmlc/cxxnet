#ifndef CXXNET_SOFTMAX_HPP
#define CXXNET_SOFTMAX_HPP
#pragma once
#include "cxxnet_layer.h"
/*!
 * \file cxxnet_softmax.hpp
 * \brief Implement of softmax layer
 * \author Bing Xu
 */

namespace cxxnet {

template<typename device, int dim>
class SoftmaxLayer: public ILayer {
public:
    /*! \brief Constructor for SoftmaxLayer
     *  \param in input node
     *  \param target target node for tarining
     */
    explicit SoftmaxLayer(Node<device, dim> in,
                          Node<device, dim> out) {
        in_node_ = in;
        out_node_ = out;
        // Alloc W, b
    }
    ~SoftmaxLayer() {
        // Dealloc W, b
    }

    void Forwardprop(bool is_train) {
        // out_node_.in = sigmoid(W * in_node_.in + b);
    }

    void Backprop() {
        // in_node_.out = loss(out_node_.out);
    }

private:
    Node<device, dim> in_node_;
    Node<device, dim> out_node_;
    mshadow::Tensor<device, 2> W_;
    mshadow::Tensor<device, 1> b_;
}; // class SoftmaxLayer

}; // namespace cxxnet

#endif // CXXNET_SOFTMAX_HPP
