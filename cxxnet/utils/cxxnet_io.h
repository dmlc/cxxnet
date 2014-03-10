#ifndef CXXNET_IO_H
#define CXXNET_IO_H
/*!
 * \file cxxnet_io.h
 * \brief io extensions
 * \author Bing Xu
 */
#include <zlib.h>
#include "cxxnet_utils.h"
#include "mshadow/tensor_io.h"

namespace cxxnet{
    namespace utils{
        typedef mshadow::utils::IStream IStream;
        
        struct GzFile : public IStream{
        public:
            GzFile( const char *path, const char *mode ) {
                fp_ = gzopen( path, mode );
                if( fp_ == NULL ){
                    fprintf( stderr, "cannot open %s", path );
                }
                Assert( fp_ != NULL, "Failed to open file\n" );
            }
            virtual ~GzFile( void ) {
                this->Close();
            }
            virtual void Close( void ){
                if ( fp_ != NULL ){
                    gzclose( fp_ ); fp_ = NULL;
                }
            }
            virtual size_t Read( void *ptr, size_t size ){
                return gzread( fp_, ptr, size );
            }
            virtual void Write( const void *ptr, size_t size ) {
                gzwrite( fp_, ptr, size );
            }
            inline int ReadInt( void ) {
                unsigned char buf[4];
                utils::Assert( Read( buf, sizeof(buf) ) == sizeof(buf), "Failed to read an int\n");
                return int(buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3]);
            }            
            inline unsigned char ReadByte( void ) {
                unsigned char i;
                utils::Assert( Read( &i, sizeof(i) ) == sizeof(i), "Failed to read an byte");
                return i;
            }
        private:
            gzFile fp_;
        };
    }
};
#endif

