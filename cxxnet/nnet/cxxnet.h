#ifndef CXXNET_H
#define CXXNET_H
#pragma once
/*!
 * \file cxxnet.h
 * \brief trainer abstraction
 * \author Bing Xu, Tianqi Chen
 */
#include <vector>
#include "mshadow/tensor.h"
#include "mshadow/tensor_io.h"
#include "../io/cxxnet_data.h"

namespace cxxnet {
    /*! \brief interface for network */
    class INetTrainer{
    public:
        // set model parameters, call this before everything, including load model
        virtual void SetParam( const char *name, const char *val ) = 0;
        // random initalize model
        virtual void InitModel( void ) = 0;
        // tell trainer which round it is
        virtual void StartRound( int epoch ) = 0;
        // save model to stream
        virtual void SaveModel( mshadow::utils::IStream &fo ) const = 0;
        // load model from stream
        virtual void LoadModel( mshadow::utils::IStream &fi ) = 0;
        // update model parameter
        virtual void Update( const DataBatch& data ) = 0;
        // evaluate a test statistics, output results into fo
        virtual void Evaluate( FILE *fo, IIterator<DataBatch> *iter_eval, const char* evname ) = 0;
        // predict labels
        virtual void Predict( std::vector<float> &preds, const DataBatch& batch ) = 0;
    };
};

namespace cxxnet {
    /*! 
     * \brief create a net implementation 
     * \param net_type network type, used to select trainer variants
     * \param device device type
     */
    INetTrainer* CreateNet( int net_type, const char *device );
};
#endif // CXXNET_H
