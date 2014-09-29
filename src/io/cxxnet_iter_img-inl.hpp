#ifndef CXXNET_ITER_IMG_INL_HPP
#define CXXNET_ITER_IMG_INL_HPP
#pragma once
/*!
 * \file cxxnet_iter_img-inl.hpp
 * \brief implementation of image iterator
 * \author Tianqi Chen
 */
// use opencv for image loading


#include "mshadow/tensor_container.h"
#include "cxxnet_data.h"
#include <opencv2/opencv.hpp>

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
#endif
