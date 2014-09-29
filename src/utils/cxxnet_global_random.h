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
#include <cmath>
#include "cxxnet_utils.h"

namespace cxxnet{
    namespace utils{
        /*! \brief return a real number uniform in [0,1) */
        inline double NextDouble(){
            return static_cast<double>( rand() ) / (static_cast<double>( RAND_MAX )+1.0);
        }
        /*! \brief return a real numer uniform in (0,1) */
        inline double NextDouble2(){
            return (static_cast<double>( rand() ) + 1.0 ) / (static_cast<double>(RAND_MAX) + 2.0);
        }        
        /*! \brief return a random number in n */
        inline uint32_t NextUInt32( uint32_t n ){
            return (uint32_t) floor( NextDouble() * n ) ;
        }

        /*! \brief return  x~N(0,1) */
        inline double SampleNormal(){
            double x,y,s;
            do{
                x = 2 * NextDouble2() - 1.0;
                y = 2 * NextDouble2() - 1.0;
                s = x*x + y*y;
            }while( s >= 1.0 || s == 0.0 );
            
            return x * sqrt( -2.0 * log(s) / s ) ;
        }
                
        /*! \brief  return distribution from Gamma( alpha, beta ) */
        inline double SampleGamma( double alpha, double beta ) {
            if ( alpha < 1.0 ) {
                double u;
                do {
                    u = NextDouble();
                } while (u == 0.0);
                return SampleGamma(alpha + 1.0, beta) * pow(u, 1.0 / alpha);
            } else {
                double d,c,x,v,u;
                d = alpha - 1.0/3.0;
                c = 1.0 / sqrt( 9.0 * d );
                do {
                    do {
                        x = SampleNormal();
                        v = 1.0 + c*x;
                    } while ( v <= 0.0 );
                    v = v * v * v;
                    u = NextDouble();
                } while ( (u >= (1.0 - 0.0331 * (x*x) * (x*x)))
                          && (log(u) >= (0.5 * x * x + d * (1.0 - v + log(v)))) );
                return d * v / beta;
            }
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

