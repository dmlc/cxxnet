#ifndef CXXNET_H
#define CXXNET_H
#pragma once
/*!
 * \file cxxnet.h
 * \brief Base definition for cxxnet
 * \author Bing Xu
 */
#include <vector>
#include "mshadow/tensor.h"
#include "mshadow/tensor_io.h"

/*! \brief namespace of cxxnet */
namespace cxxnet {
    typedef mshadow::cpu cpu;
    typedef mshadow::gpu gpu;
    typedef mshadow::index_t index_t;
    typedef mshadow::real_t  real_t;
};

namespace cxxnet {
    /*! \brief interface for network */
    class INetTrainer{
    public:
        // set model parameters
        virtual void SetParam( const char *name, const char *val ) = 0;
        // random initalize model
        virtual void InitModel( void ) = 0;
        // save model to stream
        virtual void SaveModel( mshadow::utils::IStream &fo ) const = 0;
        // load model from stream
        virtual void LoadModel( mshadow::utils::IStream &fi ) = 0;
        // update model parameter
        virtual void Update ( const std::vector<int> &labels, const mshadow::Tensor<cpu,4> &batch ) = 0;
        // predict labels
        virtual void Predict( const std::vector<int> &labels, const mshadow::Tensor<cpu,4> &batch ) = 0;
    };
};

namespace cxxnet {
    // todo 
    INetTrainer* CreateNet( int net_type );
};
#endif // CXXNET_H
