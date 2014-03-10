#ifndef CXXNET_DATA_H
#define CXXNET_DATA_H
#pragma once
/*!
 * \file cxxnet.h
 * \brief data type and iterator abstraction
 * \author Bing Xu, Tianqi Chen
 */

#include <vector>
#include <string>
#include "mshadow/tensor.h"

/*! \brief namespace of cxxnet */
namespace cxxnet {
    typedef mshadow::cpu cpu;
    typedef mshadow::gpu gpu;
    typedef mshadow::index_t index_t;
    typedef mshadow::real_t  real_t;
};

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
    
    /*! \brief a standard batch of data commonly used by iterator */
    struct DataBatch{
        /*! \brief label information */
        float*  labels;
        /*! \brief unique id for instance, can be NULL, sometimes is useful */
        unsigned* inst_index;
        /*! \brief content of data */
        mshadow::Tensor<cpu,4> data;
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
