#ifndef CXXNET_NODE_H
#define CXXNET_NODE_H
#pragma once
/*!
 * \file cxxnet_node_h
 * \brief Data structure for saveing data in the network
 * \author Tianqi Chen, Bing, Xu
 */

#include "cxxnet.h"

namespace cxxnet {

/*! \brief abstruct class for Node */
template<typename device, int dim>
class Node {
public:
    /*!
     * \brief constructor for Node class
     * \param mshadow::Tensor
     * \tparam device cpu or gpu
     * \tparam dim dimention of the node, 2D or 4D
     */
    // Use explict way to init with 2D or 4D?
    Node(mshadow::Tensor<device, dim> t) { in_data_ = t; }
    /*! \brief get input tensor */
    inline mshadow::Tensor<device, dim> GetInputTensor() { return in_data_; }
    /*! \brief get output tensor */
    inline mshadow::Tensor<device, dim> GetOutputTensor() { return out_data_; }
private:
    mshadow::Tensor in_data_;
    mshadow::Tensor out_data_;
}; // class Node

}; // namepsace cxxnet
#endif // CXXNET_NODE_H
