#ifndef CXXNET_IO_UTILS_H
#define CXXNET_IO_UTILS_H
/*!
 * \file cxxnet_io_utils.h
 * \brief io extensions
 * \author Bing Xu
 */
#include <zlib.h>
#include "cxxnet_utils.h"
#include "mshadow/tensor_io.h"

namespace cxxnet{
    namespace utils{
        typedef mshadow::utils::IStream IStream;

        /*!
         * \brief interface of stream that containes seek option,
         *   mshadow does not rely on this interface
         *   this is not always supported(e.g. in socket)
         */
        class ISeekStream : public IStream{
        public:
            /*!
             * \brief seek to a position, warning:
             * \param pos relative position to start of stream
             */
            virtual void Seek( size_t pos ){
                utils::Error("Seek is not implemented");
            }
        public:
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
        };

        struct GzFile : public ISeekStream{
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
            virtual void Seek( size_t pos ){
                gzseek( fp_, pos, SEEK_SET );
            }
        private:
            gzFile fp_;
        };

        /*! \brief implementation of file i/o stream */
        class StdFile: public ISeekStream{
        public:
            /*! \brief constructor */
            StdFile( const char *fname, const char *mode ){
                fp_ = utils::FopenCheck( fname, mode );
            }
            virtual ~StdFile( void ){
                this->Close();
            }
            virtual size_t Read( void *ptr, size_t size ){
                return fread( ptr, size, 1, fp_ );
            }
            virtual void Write( const void *ptr, size_t size ){
                fwrite( ptr, size, 1, fp_ );
            }
            virtual void Seek( size_t pos ){
                fseek( fp_, pos, SEEK_SET );
            }
            inline void Close( void ){
                if ( fp_ != NULL ){
                    fclose( fp_ ); fp_ = NULL;
                }
            }
        private:
            FILE *fp_;
        };
    };
};

namespace cxxnet {
    /*! \brief Basic page class */
    class BinaryPage {
    public:
        /*! \brief page size */
        static const size_t kPageSize = 1 << 24;
    public:
        /*! \brief memory data object */
        struct Obj{
            /*! \brief pointer to the data*/
            void  *dptr;
            /*! \brief size */
            size_t sz;
            Obj( void * dptr, size_t sz ):dptr(dptr),sz(sz){}
        };
    public:
        /*! \brief constructor of page */
        BinaryPage( void ):nelem_(data_[0]){
            this->Clear();
        };
        /*! 
         * \brief load one page form instream
         * \return true if loading is successful
         */
        inline bool Load( utils::IStream &fi) {
            return fi.Read(&data_[0], sizeof(int)*kPageSize ) !=0;
        }
        /*! \brief save one page into outstream */
        inline void Save( utils::IStream &fo ) {
            fo.Write( &data_[0], sizeof(int)*kPageSize );
        }
        /*! \return number of elements */
        inline int nelem( void ){
            return nelem_;
        }
        /*! \brief Push one binary object into page
         *  \param fname file name of obj need to be pushed into
         *  \return false or true to push into
         */
        inline bool Push( const Obj &dat ) {
            if( this->FreeBytes() < dat.sz + sizeof(int) ) return false;
            data_[ nelem_ + 2 ] = data_[ nelem_ + 1 ] + dat.sz;
            memcpy( this->offset( data_[ nelem_ + 2 ]), dat.dptr, dat.sz );
            ++ nelem_;
            return true;
        }
        /*! \brief Clear the page */
        inline void Clear( void ) {
            memset( &data_[0], 0, sizeof(int) * kPageSize );            
        }
        /*! 
         * \brief Get one binary object from page
         *  \param r r th obj in the page
         *  \param obj BinaryObj struct to save the obj info
         */
        inline Obj operator[]( int r ){
            utils::Assert( r < nelem(), "index excceed bound" );
            return Obj( this->offset( data_[ r + 2 ] ),  data_[ r + 2 ] - data_[ r + 1 ] );
        }
    private:
        /*! \return number of elements */
        inline size_t FreeBytes( void ){
            return ( kPageSize - (nelem_ + 2) ) * sizeof(int) - data_[ nelem_ + 1 ];
        }
        inline void* offset( int pos ){
            return (char*)(&data_[0]) + (kPageSize*sizeof(int) - pos);
        }
    private:
        int data_[ kPageSize ];
        int &nelem_;
    }; // class BinaryPage
}; // namespace cxxnet
#endif
