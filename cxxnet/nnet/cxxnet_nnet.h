#ifndef CXXNET_NNET_H
#define CXXNET_NNET_H
#pragma once
/*!
 * \file cxxnet_nnet.h
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
        virtual ~INetTrainer( void ){}
        /*! \brief set model parameters, call this before everything, including load model */
        virtual void SetParam( const char *name, const char *val ) = 0;
        /*! \brief random initalize model */
        virtual void InitModel( void ) = 0;
        /*! \brief save model to stream */
        virtual void SaveModel( mshadow::utils::IStream &fo ) const = 0;
        /*! \brief load model from stream */
        virtual void LoadModel( mshadow::utils::IStream &fi ) = 0;
        /*!
         * \brief inform the updater that a new round has been started
         * \param round round counter
         */
        virtual void StartRound( int round ) = 0;
        /*!
         * \brief update model parameter
         * \param training data batch
         */
        virtual void Update( const DataBatch& data ) = 0;
        /*! \brief  evaluate a test statistics, output results into fo */
        virtual void Evaluate( FILE *fo, IIterator<DataBatch> *iter_eval, const char* evname ) = 0;
        /*! \brief  predict labels */
        virtual void Predict( std::vector<float> &preds, const DataBatch& batch ) = 0;
    };
};

namespace cxxnet {
    /*!
     * \brief create a CPU net implementation
     * \param net_type network type, used to select trainer variants
     */
    INetTrainer* CreateNetCPU( int net_type );
    /*!
     * \brief create a GPU net implementation
     * \param net_type network type, used to select trainer variants
     * \param devid device id
     */
    INetTrainer* CreateNetGPU( int net_type );
    /*!
     * \brief create a net implementation
     * \param net_type network type, used to select trainer variants
     * \param device device type
     */
    inline INetTrainer* CreateNet( int net_type, const char *device ){
        if( !strcmp( device, "cpu") ) return CreateNetCPU( net_type );
        if( !strcmp( device, "gpu") ) return CreateNetGPU( net_type );
        utils::Error("unknown device type" );
        return NULL;
    }
};
#endif // CXXNET_H
