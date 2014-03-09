#ifndef CXXNET_DATA_H
#define CXXNET_DATA_H
#pragma once

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
        /*! \brief get current matrix */
        virtual const DType &value( void ) const = 0;
    public:
        /*! \brief constructor */
        virtual ~IIterator( void ){}
    };
};
#endif
