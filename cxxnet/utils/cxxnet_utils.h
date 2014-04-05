#ifndef CXXNET_UTILS_H
#define CXXNET_UTILS_H
/*!
 * \file cxxnet_utils.h
 * \brief simple utils to support the code, adapted from xgboost
 * \author Tianqi Chen
 */

#define _CRT_SECURE_NO_WARNINGS
#ifdef _MSC_VER
#define fopen64 fopen
#else

// use 64 bit offset, either to include this header in the beginning, or
#ifdef _FILE_OFFSET_BITS
#if _FILE_OFFSET_BITS == 32
#warning "FILE OFFSET BITS defined to be 32 bit"
#endif
#endif

#ifdef __APPLE__
#define off64_t off_t
#define fopen64 fopen
#endif

#define _FILE_OFFSET_BITS 64
extern "C"{
#include <sys/types.h>
};
#include <cstdio>
#endif

#ifdef _MSC_VER
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int  uint32_t;
#else
#include <inttypes.h>
#endif

#include <cstdio>
#include <cstdlib>

namespace cxxnet{
    /*! \brief namespace for helper utils of the project */
    namespace utils{
        inline void Error( const char *msg ){
            fprintf( stderr, "Error:%s\n",msg );
            exit( -1 );
        }

        inline void Assert( bool exp ){
            if( !exp ) Error( "AssertError" );
        }

        inline void Assert( bool exp, const char *msg ){
            if( !exp ) Error( msg );
        }

        inline void Warning( const char *msg ){
            fprintf( stderr, "warning:%s\n",msg );
        }

        /*! \brief replace fopen, report error when the file open fails */
        inline FILE *FopenCheck( const char *fname , const char *flag ){
            FILE *fp = fopen64( fname , flag );
            if( fp == NULL ){
                fprintf( stderr, "can not open file \"%s\"\n",fname );
                exit( -1 );
            }
            return fp;
        }
    };
};
#endif
