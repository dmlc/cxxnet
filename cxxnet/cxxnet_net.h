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
    // all the interfaces
    /*! \brief update algorithm that defines parameter updates */
    class IUpdater{
    public:
        /*! \brief update parameter */
        virtual void Update( void ) = 0;
    };

    /*! \brief interface of layer */
    class ILayer {
    public:
        /*!
         * \brief Forward propagation from in_node to out_node
         * \param is_train the propagation is training or dropout
         */
        virtual void Forwardprop(bool is_train) = 0;
        /*!
         * \brief Back propagation from out_node to in_node, generate the gradient, out_node already stores gradient value
         * \param is_firstlayer if true, then the layer will not propagate gradient back to its input node
         */
        virtual void Backprop(bool is_firstlayer) = 0;
        /*!
         * \brief Get updaters for the layer
         * \param updaters updater for the whole network
         */
        virtual void GetUpdaters(std::vector<IUpdater*> &updaters) = 0;
        /*!
         * \brief Set param for the layer from string
         * \param name parameter name
         * \param val string for configuration
         */
        virtual void SetParam(const char *name, const char* val) = 0;
        /*!
         * \brief intialized model parameters
         */        
        virtual void InitModel( void ) = 0;
        /*!
         * \brief Save model into binary file
         * \param fo output stream
         */
        virtual void SaveModel(mshadow::utils::IStream &fo) const = 0;
        /*!
         * \brief Load model from binary file
         * \param fi input stream
         */
        virtual void LoadModel(mshadow::utils::IStream &fi)= 0;
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

    /*! \brief potential parameters for each layer */
    struct LayerParam{
        /*! \brief learning rate */
        float eta;
        /*! \brief mometum */
        float momentum;
        /*! \brief initialization sd for weight */
        float init_sigma;
        /*! \brief updater */        
        std::string updater;
        /*!
         * \brief Set param for the layer from string
         * \param name parameter name
         * \param val string for configuration
         */
        virtual void SetParam(const char *name, const char* val) {
            if( !strcmp( name, "eta") )           eta = (float)atof(val);
            if( !strcmp( name, "learning_rate") ) eta = (float)atof(val);
            if( !strcmp( name, "momentum") )      momentum = (float)atof(val);
            if( !strcmp( name, "updater") )       updater = val;            
        }
    };
}; // namespace cxxnet

namespace cxxnet {
    /*! 
     * \brief factory: create an upadater algorithm of given type
     *        to be decided, maybe need other abstraction
     * \param lparam parameter of each layer
     * \param weight network weight
     * \param grad network gradient 
     */
    template<typename xpu, int dim>
    inline IUpdater* CreateUpdater( LayerParam &lparam, 
                                    mshadow::Tensor<xpu,dim> &weight, 
                                    const mshadow::Tensor<xpu,dim> &grad );    
};  // namespace cxxnet

#include "cxxnet_layer-inl.hpp"

#endif // CXXNET_LAYER_H
