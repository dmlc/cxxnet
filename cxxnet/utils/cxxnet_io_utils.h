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
    /*! \brief struct to save binary object in page */
    struct BinaryObj {
        char *ptr_;
        size_t sz_;
    }; // struct BinaryObj

    /*! \brief Basic page class
     *  \tparam basic element size: int or size_t
     */
    template<typename T>
    class BinaryPage {
    public:
        // page size
        static const size_t psize = 1 << 24;
    private:
        T *dptr_;
        T *head_;
        T *tail_;
        T now_;
    public:
        /*! \brief constructor of page */
        BinaryPage() {
            dptr_ = NULL;
            dptr_ = new T[psize];
            utils::Assert(dptr_ != NULL);
            dptr_[0] = 0;
            now_ = 0;
            head_ = dptr_ + 1;
            tail_ = dptr_ + psize;
        };

        /*! deconstructor of page */
        ~BinaryPage() { if (dptr_) delete [] dptr_; }
        /*! \brief load one page form instream */
        inline void Load(FILE *fi) {
            utils::Assert(fread(dptr_, sizeof(T), psize, fi) > 0);
        }
        /*! \brief save one page into outstream */
        inline void Save(FILE *fo) {
            fwrite(dptr_, sizeof(T), psize, fo);
        }
        /*! \brief Push one binary object into page
         *  \param fname file name of obj need to be pushed into
         *  \return false or true to push into
         */
        inline bool Push(const char * fname) {
            // get file length
            FILE* fp = fopen(fname, "rb");
            utils::Assert(fp != NULL);
            fseek(fp, 0L, SEEK_END);
            size_t sz = ftell(fp);
            fseek(fp, 0L, SEEK_SET);
            bool success = false;

            // judge whether can set in
            if ((char*)(head_ + 2) < ((char *)tail_ - sz)) {
                dptr_[0] ++;
                head_[now_++] = sz;
                char *pstart = (char *)tail_ - sz;
                tail_ = (T*) (pstart - 1);
                fread(pstart, sizeof(char), sz, fp);
                success = true;
            }
            fclose(fp);
            return success;
        }
        /*! \brief Get total object in the page */
        inline T Size() { return dptr_[0]; }

        /*! \brief Clear the page */
        inline void Clear() {
            dptr_[0] = 0;
            head_ = dptr_ + 1;
            tail_ = dptr_ + psize;
            now_ = 0;
        }

        /*! \brief Get one binary object from page
         *  \param r r th obj in the page
         *  \param obj BinaryObj struct to save the obj info
         */
        inline bool Get(int r, BinaryObj &obj) {
            if (r > dptr_[0]) return false;
            obj.sz_ = dptr_[r];
            obj.ptr_ = (char*)(dptr_ + psize);
            for (int i = 0; i <= r; ++i) {
                obj.ptr_ -= dptr_[1 + i];
                if (i != r) obj.ptr_ -= 1;
            }
            return true;
        }
    }; // class BinaryPage

    template<typename T>
    class BinaryPageWriter {
    private:
        BinaryPage<T> pg_;
        FILE *fp_;
    public:
        BinaryPageWriter(const char *name) {
            fp_ = fopen(name, "wb");
            utils::Assert(fp_ != NULL);
        }
        void Push(const char *name) {
            if (!pg_.Push(name)) {
                pg_.Save(fp_);
                pg_.Clear();
            }
            if (!pg_.Push(name)) {
                utils::Error("Single Page too small");
            }
        }
        ~BinaryPageWriter() {
            if (fp_) fclose(fp_);
        }
    }; // class BinaryPageWriter


    template<typename T>
    class BinaryPageReader {
    private:
        BinaryPage<T> pg_;
        FILE *fp_;
    public:
        int now_elem_;
        int now_page_;
        size_t total_page_;
        size_t max_obj_;
    public:
        BinaryPageReader(const char *name) {
            fp_ = fopen(name, "rb");
            utils::Assert(fp_ != NULL, "Can not open the page file!");
            total_page_ = ftell(fp_) / pg_.psize / sizeof(T);
            this->Reset();
            max_obj_ = total_page_ * pg_.Size();
        }
        inline void Reset() {
            now_elem_ = 0;
            now_page_ = 0;
            fseek(fp_, 0L, SEEK_SET);
            pg_.Load(fp_);
        }
        inline bool Next(BinaryObj &obj) {
            if (now_page_ < total_page_) {
                if (!pg_.Get(now_elem_++, obj)) {
                    if (now_page_ + 1 < total_page_) {
                        pg_.Load(fp_);
                        now_page_++;
                        now_elem_ = 0;
                        return pg_.Get(now_elem_++, obj);
                    } else {
                        return false;
                    }
                }
                return true;
            }
            return false;
        }
        inline bool Get(size_t r, BinaryObj &obj) {
            utils::Error("Not implemented");
            return false;
        }
    }; //class BinaryPageWriter

}; // namespace cxxnet
#endif

