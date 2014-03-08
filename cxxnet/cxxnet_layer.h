#ifndef CXXNET_LAYER_H
#define CXXNET_LAYER_H
#pragma once
/*!
 * \file cxxnet_layer.h
 * \brief Abstruct definition for layer interface
 * \author Tianqi Chen, Bing Xu
 */

#include "cxxnet.h"
namespace cxxnet {

class ILayer {
public:
    /*!
     * \brief Forward propagation from in_node to out_node
     * \param is_train the propagation is training or dropout
     */
    virtual void Forwardprop(bool is_train) = 0;
    /*!
     * \brief Back propagation from out_node to in_node, generate the gradient
     */
    virtual void Backprop() = 0;
    /*!
     * \brief Get updaters for the layer
     * \param updaters updater for the whole network
     */
    virtual void GetUpdaters(std::vector<IUpdaterAlgo*> &updaters) = 0;
    /*!
     * \brief Set param for the layer from string
     * \param name parameter name
     * \param val string for configuration
     */
    virtual void SetParam(const char *name, const char* val) = 0;
    /*!
     * \brief Save model into binary file
     * \param fo output stream
     */
    virtual void SaveModel(mshadow::utils::Stream &fo) const = 0;
    /*!
     * \brief Load model from binary file
     * \param fi input stream
     */
    virtual void LoadModel(mshadow::utils::Stream &fi) const = 0;
}; // namespace cxxnet

#endif // CXXNET_LAYER_H
