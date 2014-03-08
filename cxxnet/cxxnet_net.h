#ifndef CXXNET_NET_H
#define CXXNET_NET_H
#pragma once
/*!
 * \file cxxnet_net.h
 * \brief Abstruct definition for layer interface, 
 *        data type, everything used to construct a network
 * \author Tianqi Chen, Bing Xu
 */
#include <vector>
#include <string>
#include "cxxnet.h"
#include "mshadow/tensor.h"
#include "mshadow/tensor_io.h"

namespace cxxnet {
    /*! \brief interface of a updater */
    class IUpdater{
    public:
        /*!\brief virtual destructor */
        virtual ~IUpdater( void ){}
        /*! \brief update parameter */
        virtual void Update( void ) = 0;
        /*!\ brief set parameters that could be spefic to this updater */
        virtual void SetParam( const char *name, const char *val ) = 0;
    };

    /*! \brief interface of layer */
    class ILayer {
    public:
        /*!\brief virtual destructor */
        virtual ~ILayer( void ){}
        /*!
         * \brief Forward propagation from in_node to out_node
         * \param is_train the propagation is training or dropout
         */
        virtual void Forward(bool is_train) = 0;
        /*!
         * \brief Back propagation from out_node to in_node, generate the gradient, out_node already stores gradient value
         * \param is_firstlayer if true, then the layer will not propagate gradient back to its input node
         */
        virtual void Backprop(bool is_firstlayer) = 0;
    public:
        // interface code that not needed to be implemented by all nodes
        /*!
         * \brief Get updaters for the layer
         * \param specified updater type 
         * \param updaters the laeyer will push_back into updaters 
         */
        virtual void GetUpdaters( const char *updater, std::vector<IUpdater*> &updaters ) {}
        /*!
         * \brief Set param for the layer from string
         * \param name parameter name
         * \param val string for configuration
         */
        virtual void SetParam(const char *name, const char* val) {}
        /*!
         * \brief intialized model parameters
         */        
        virtual void InitModel(void) {}
        /*!
         * \brief Save model into binary file
         * \param fo output stream
         */
        virtual void SaveModel(mshadow::utils::IStream &fo) const {}
        /*!
         * \brief Load model from binary file
         * \param fi input stream
         */
        virtual void LoadModel(mshadow::utils::IStream &fi) {}
    }; 
}; // namespace cxxnet

namespace cxxnet {
    // potentially useful data structures
    /*! \brief abstruct class for Node */
    template<typename xpu>
    struct Node {
        /*! \brief content of the node */
        mshadow::Tensor<xpu,4> data;
        /*! \brief matrix view of the node */
        inline mshadow::Tensor<xpu,2> mat( void ){
            return data[0][0];
        }        
    }; // struct Node 
}; // namespace cxxnet

namespace cxxnet {
    /*! 
     * \brief factory: create an upadater algorithm of given type
     * \param type indicate the type of updater
     * \param rnd random number generator
     * \param weight network weight
     * \param grad network gradient 
     * \param tag some tags used to identify the weight, for example: "bias", "wmat", "mask", default ""
     */
    template<typename xpu, int dim>
    inline IUpdater* CreateUpdater( const char *type,
                                    mshadow::Random<xpu> &rnd, 
                                    mshadow::Tensor<xpu,dim> &weight, 
                                    mshadow::Tensor<xpu,dim> &wgrad,
                                    const char *tag );

    /*! 
     * \brief factory: create an upadater algorithm of given type
     * \param type indicate the type of a layer
     * \param rnd random number generator
     * \param in input node
     * \param out output node
     */
    template<typename xpu>
    inline ILayer* CreateLayer( const char *type, mshadow::Random<xpu> &rnd, Node<xpu>& in, Node<xpu>& out );
};  // namespace cxxnet

#include "cxxnet_updater-inl.hpp"
#include "cxxnet_layer-inl.hpp"

#endif // CXXNET_NET_H
