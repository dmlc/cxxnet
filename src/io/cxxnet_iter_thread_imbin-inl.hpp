#ifndef CXXNET_ITER_THREAD_IMBIN_INL_HPP
#define CXXNET_ITER_THREAD_IMBIN_INL_HPP
#pragma once
/*!
 * \file cxxnet_iter_thread_imbin-inl.hpp
 * \brief threaded version of page iterator
 * \author Tianqi Chen
 */
#include "cxxnet_data.h"
#include <opencv2/opencv.hpp>
#include "../utils/cxxnet_thread_buffer.h"
#include "../utils/cxxnet_io_utils.h"

namespace cxxnet{
    /*! \brief thread buffer iterator */
    class ThreadImagePageIterator: public IIterator< DataInst >{
    public:
        ThreadImagePageIterator( void ){
            img_.set_pad( false );
            fplst_ = NULL;
            silent_ = 0;
            path_imglst_ = "img.lst";
            path_imgbin_ = "img.bin";
            itr.SetParam( "buffer_size", "4" );
            page_.page = NULL;
        }
        virtual ~ThreadImagePageIterator( void ){
            if( fplst_ != NULL ) fclose( fplst_ );
        }
        virtual void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "image_list" ) )    path_imglst_ = val;
            if( !strcmp( name, "image_bin") )     path_imgbin_ = val;
            if( !strcmp( name, "silent"   ) )      silent_ = atoi( val );
        }
        virtual void Init( void ){
            fplst_  = utils::FopenCheck( path_imglst_.c_str(), "r" );
            if( silent_ == 0 ){
                printf("ThreadImagePageIterator:image_list=%s, bin=%s\n", path_imglst_.c_str(), path_imgbin_.c_str() );
            }
            itr.get_factory().fi.Open( path_imgbin_.c_str(), "rb" );
            itr.Init();
            this->BeforeFirst();
        }
        virtual void BeforeFirst( void ){
            fseek( fplst_ , 0, SEEK_SET );
            itr.BeforeFirst();
            this->LoadNextPage();
        }
        virtual bool Next( void ){
            while( fscanf( fplst_,"%u%f%*[^\n]\n", &out_.index, &out_.label ) == 2 ){
                this->NextBuffer( buf_ );
                this->LoadImage( img_, out_, buf_ );
                return true;
            }
            return false;
        }
        virtual const DataInst &Value( void ) const{
            return out_;
        }
    protected:
        inline static void LoadImage( mshadow::TensorContainer<cpu,3> &img,
                                      DataInst &out,
                                      std::vector<unsigned char>& buf ){
            cv::Mat res = cv::imdecode( buf, 1 );
            utils::Assert( res.data != NULL, "decoding fail" );

            img.Resize( mshadow::Shape3( 3, res.rows, res.cols ) );
            for( index_t y = 0; y < img.shape[1]; ++y ){
                for( index_t x = 0; x < img.shape[0]; ++x ){
                    cv::Vec3b bgr = res.at<cv::Vec3b>( y, x );
                    // store in RGB order
                    img[2][y][x] = bgr[0];
                    img[1][y][x] = bgr[1];
                    img[0][y][x] = bgr[2];
                }
            }
            out.data = img;
            // free memory
            res.release();
        }
        inline void NextBuffer( std::vector<unsigned char> &buf ){
            while( ptop_ >= page_.page->Size() ){
                this->LoadNextPage();
            }
            utils::BinaryPage::Obj obj = (*page_.page)[ ptop_ ];
            buf.resize( obj.sz );
            memcpy( &buf[0], obj.dptr, obj.sz );
            ++ ptop_;
        }
        inline void LoadNextPage( void ){
            utils::Assert( itr.Next( page_ ), "can not get first page" );
            ptop_ = 0;
        }
    protected:
        // output data
        DataInst out_;
        // silent
        int silent_;
        // file pointer to list file, information file
        FILE *fplst_;
        // prefix path of image binary, path to input lst, format: imageid label path
        std::string path_imgbin_, path_imglst_;
        // temp storage for image
        mshadow::TensorContainer<cpu,3> img_;
        // temp memory buffer
        std::vector<unsigned char> buf_;
    private:
        struct PagePtr{
            utils::BinaryPage *page;
        };
        struct Factory{
        public:
            utils::StdFile fi;
        public:
            Factory(){}
            inline bool Init(){
                return true;
            }
            inline void SetParam( const char *name, const char *val ){}
            inline bool LoadNext( PagePtr &val ){
                return val.page->Load( fi );
            }
            inline PagePtr Create( void ){
                PagePtr a; a.page = new utils::BinaryPage();
                return a;
            }
            inline void FreeSpace( PagePtr &a ){
                delete a.page;
            }
            inline void Destroy(){
            }
            inline void BeforeFirst(){
                fi.Seek( 0 );
            }
        };
    protected:
        PagePtr page_;
        int     ptop_;
        utils::ThreadBuffer<PagePtr,Factory> itr;
    };
};
#endif

