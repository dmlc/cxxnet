#ifndef CXXNET_ITER_IMG_INL_HPP
#define CXXNET_ITER_IMG_INL_HPP
#pragma once
/*!
 * \file cxxnet_iter_img-inl.hpp
 * \brief implementation of image iterator
 * \author Tianqi Chen
 */
#include "mshadow/tensor_container.h"
#include "cxxnet_data.h"
// use opencv for image loading
#include <opencv2/opencv.hpp>
#include <omp.h>

namespace cxxnet{
    /*! \brief simple image iterator that only loads data instance */
    class ImageIterator : public IIterator< DataInst >{
    public:
        ImageIterator( void ){
            img_.set_pad( false );
            fplst_ = NULL;
            silent_ = 0;
            path_imgdir_ = "";
            path_imglst_ = "img.lst";
        }
        virtual ~ImageIterator( void ){
            if( fplst_ != NULL ) fclose( fplst_ );
        }
        virtual void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "image_list" ) )    path_imglst_ = val;
            if( !strcmp( name, "image_root") )     path_imgdir_ = val;
            if( !strcmp( name, "silent"   ) )      silent_ = atoi( val );
        }
        virtual void Init( void ){
            fplst_  = utils::FopenCheck( path_imglst_.c_str(), "r" );
            if( silent_ == 0 ){
                printf("ImageIterator:image_list=%s\n", path_imglst_.c_str() );
            }
            this->BeforeFirst();
        }
        virtual void BeforeFirst( void ){
            fseek( fplst_ , 0, SEEK_SET );
        }
        virtual bool Next( void ){
            char fname[ 256 ], sname[256];
            while( fscanf( fplst_,"%u%f %[^\n]\n", &out_.index, &out_.label, fname ) == 3 ){
                if( fname[0] == '\0' ) continue;
                if( path_imgdir_.length() == 0 ){
                    LoadImage( img_, out_, fname );
                }else{
                    sprintf( sname, "%s%s", path_imgdir_.c_str(), fname );
                    LoadImage( img_, out_, sname );
                }
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
                                      const char *fname ){
            cv::Mat res = cv::imread( fname );
            if( res.data == NULL ){
                fprintf( stderr, "LoadImage: image %s not exists\n", fname );
                utils::Error( "LoadImage: image not exists" );
            }
            img.Resize( mshadow::Shape3( 3, res.rows, res.cols ) );
            for( index_t z = 0; z < img.shape[2]; ++z ){
                for( index_t y = 0; y < img.shape[1]; ++y ){
                    for( index_t x = 0; x < img.shape[0]; ++x ){
                        img[0][y][x] = res.data[ x * res.step + y + z ];
                        img[1][y][x] = res.data[ x * res.step + y + z ];
                        img[2][y][x] = res.data[ x * res.step + y + z ];
                    }
                }
            }
            out.data = img;
        }
    protected:
        // output data
        DataInst out_;
        // silent
        int silent_;
        // file pointer to list file, information file
        FILE *fplst_;
        // prefix path of image folder, path to input lst, format: imageid label path
        std::string path_imgdir_, path_imglst_;
        // temp storage for image
        mshadow::TensorContainer<cpu,3> img_;
    };
};

namespace cxxnet{
    /*! \brief openmp image iterator*/
    class OMPImageIterator : public ImageIterator{
    public:
        OMPImageIterator( void ){
            nthreads_ = 4;
            buf_size_ = 64;
        }
        virtual void SetParam( const char *name, const char *val ){
            ImageIterator::SetParam( name, val );
            if( !strcmp( name, "image_omp_buffer") )  buf_size_ = (unsigned)atoi(val);
            if( !strcmp( name, "image_omp_nthread") ) nthreads_ = atoi(val);
        }
        virtual void Init( void ){
            ImageIterator::Init();
            for( unsigned i = 0; i < buf_size_; ++ i ){
                buf_img_.push_back( mshadow::TensorContainer<cpu,3>( false ) );
            }
            buf_out_.resize( buf_size_ );
            buf_fnames_.resize( buf_size_ );
            if( silent_ == 0 ){
                printf("OmpIterator:nthreads = %d\n", nthreads_ );
            }
        }        
        virtual bool Next( void ){
            if( buf_top_ >= buf_readed_ ){
                this->LoadBuffer();
                if( buf_readed_ == 0 ) return false;
            }
            this->out_ = buf_out_[ buf_top_++ ];
            return true;
        }
    private:        
        inline void LoadBuffer( void ){
            char fname[256], sname[256];
            buf_top_ = 0; buf_readed_ = 0;
            while( fscanf( fplst_,"%u%f %[^\n]\n", &buf_out_[buf_readed_].index, &buf_out_[buf_readed_].label, fname ) == 3 ){
                if( fname[0] == '\0' ) continue;
                if( path_imgdir_.length() == 0 ){
                    buf_fnames_[ buf_readed_ ] = fname;
                }else{
                    sprintf( sname, "%s%s", path_imgdir_.c_str(), fname );
                    buf_fnames_[ buf_readed_ ] = sname;
                }
                if( ++ buf_readed_ >= buf_size_ ) break; 
            }
            #pragma omp parallel for schedule(static) num_threads(nthreads_)
            for( unsigned i = 0; i < buf_readed_; ++ i ){
                ImageIterator::LoadImage( buf_img_[i], buf_out_[i], buf_fnames_[i].c_str() );
                
            }
        }
    private:
        // number of loading threads
        int nthreads_;
        // size of buffer, top of buffer
        unsigned buf_size_, buf_top_, buf_readed_;
        // temp storage for image
        std::vector< mshadow::TensorContainer<cpu,3> > buf_img_;
        std::vector< DataInst > buf_out_;
        std::vector< std::string >   buf_fnames_;
    };
};
#endif

