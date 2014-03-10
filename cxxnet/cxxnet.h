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
        // set model parameters, call this before everything, including load model
        virtual void SetParam( const char *name, const char *val ) = 0;
        // random initalize model
        virtual void InitModel( void ) = 0;
        // save model to stream
        virtual void SaveModel( mshadow::utils::IStream &fo ) const = 0;
        // load model from stream
        virtual void LoadModel( mshadow::utils::IStream &fi ) = 0;
        // update model parameter
        virtual void Update ( const std::vector<float> &labels, const mshadow::Tensor<cpu,4> &batch ) = 0;
        // predict labels
        virtual const mshadow::Tensor<cpu,2>& Predict( const mshadow::Tensor<cpu,4> &batch ) = 0;
    };
};

namespace cxxnet {
    INetTrainer* CreateNet( int net_type, const char *device );
};
#endif // CXXNET_H
