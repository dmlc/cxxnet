#ifndef CXXNET_DATA_H
#define CXXNET_DATA_H
#pragma once
/*!
 * \file cxxnet_data.h
 * \brief data type and iterator abstraction
 * \author Bing Xu, Tianqi Chen
 */

#include <vector>
#include <string>
#include "mshadow/tensor.h"
#include "../utils/cxxnet_utils.h"

namespace cxxnet {
    /*! 
     * \brief iterator type 
     * \tparam DType data type
     */
    template<typename DType>
    class IIterator{
    public:
        /*!
         * \brief set the parameter 
         * \param name name of parameter
         * \param val  value of parameter
         */        
        virtual void SetParam( const char *name, const char *val ) = 0;
        /*! \brief initalize the iterator so that we can use the iterator */
        virtual void Init( void ) = 0;       
        /*! \brief set before first of the item */
        virtual void BeforeFirst( void ) = 0;
        /*! \brief move to next item */
        virtual bool Next( void ) = 0;
        /*! \brief get current data */
        virtual const DType &Value( void ) const = 0;
    public:
        /*! \brief constructor */
        virtual ~IIterator( void ){}
    };
    
    /*! \brief a single data instance */
    struct DataInst{
        /*! \brief label information */
        float  label;
        /*! \brief unique id for instance */
        unsigned index;
        /*! \brief content of data */
        mshadow::Tensor<mshadow::cpu,3> data;
    };

    /*! \brief a standard batch of data commonly used by iterator */
    struct DataBatch{
        /*! \brief label information */
        float*  labels;
        /*! \brief unique id for instance, can be NULL, sometimes is useful */
        unsigned* inst_index;
        /*! \brief number of instance */
        mshadow::index_t batch_size;
        /*! \brief content of data */
        mshadow::Tensor<mshadow::cpu,4> data;
        /*! \brief constructor */
        DataBatch( void ){
            labels = NULL; inst_index = NULL; batch_size = 0;
        }
        /*! \brief auxiliary to allocate space, if needed */
        inline void AllocSpace( mshadow::Shape<4> shape, mshadow::index_t batch_size, bool pad = false ){
            data = mshadow::NewTensor<mshadow::cpu>( shape, 0.0f, pad );
            labels = new float[ batch_size ];
            inst_index = new unsigned[ batch_size ];
            this->batch_size = batch_size;
        }
        /*! \brief auxiliary function to free space, if needed*/
        inline void FreeSpace( void ){
            if( labels != NULL ){
                delete [] labels;
                delete [] inst_index;
                mshadow::FreeSpace( data );
                labels = NULL;
            }
        }
        /*! \brief copy content from existing data */
        inline void CopyFrom( const DataBatch &src ){
            utils::Assert( batch_size == src.batch_size );
            memcpy( labels, src.labels, batch_size * sizeof( float ) );
            memcpy( inst_index, src.inst_index, batch_size * sizeof( unsigned ) );
            mshadow::Copy( data, src.data );
        }
    };
};

namespace cxxnet {
    /*! 
     * \brief create iterator from configure settings 
     * \param cfg configure settings key=vale pair
     */
    IIterator<DataBatch>* CreateIterator( const std::vector< std::pair<std::string,std::string> > &cfg );
};
#endif
