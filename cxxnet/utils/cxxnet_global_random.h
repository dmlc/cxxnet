#ifndef CXXNET_GLOBAL_RANDOM_H
#define CXXNET_GLOBAL_RANDOM_H
#pragma once
/*!
 * \file cxxnet_global_random.h
 * \brief global random number utils, used for some preprocessing
 * \author Tianqi Chen
 */
#include <cstdlib>
#include <vector>

namespace cxxnet{
    namespace utils{
        /*! \brief return a real number uniform in [0,1) */
        inline double NextDouble(){
            return static_cast<double>( rand() ) / (static_cast<double>( RAND_MAX )+1.0);
        }
        /*! \brief return a random number in n */
        inline uint32_t NextUInt32( uint32_t n ){
            return (uint32_t) floor( NextDouble() * n ) ;
        }  
        /*! \brief random shuffle data */        
        template<typename T>
        inline static void Shuffle( T *data, size_t sz ){
            if( sz == 0 ) return;
            for( uint32_t i = (uint32_t)sz - 1; i > 0; i-- ){
                std::swap( data[i], data[ NextUInt32( i+1 ) ] );
            } 
        }
        /*!\brief random shuffle data in */
        template<typename T>
        inline static void Shuffle( std::vector<T> &data ){
            Shuffle( &data[0], data.size() );
        }
    };
};

#endif

