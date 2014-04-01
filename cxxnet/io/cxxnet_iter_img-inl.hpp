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
// use cimg for image loading
#include "../../thirdparty/CImg.h"

namespace cxxnet{
    /*! \brief simple image iterator that only loads data instance */
    class ImageIterator : public IIterator< DataInst >{
    public:
        ImageIterator( void ){
            img_.set_pad( false );
            fplst_ = NULL;
            silent_ = 0;
            grey_scale_ = 0;
            path_imgdir_ = "";
            path_imglst_ = "img.lst";
        }
        virtual ~ImageIterator( void ){
            if( fplst_ != NULL ) fclose( fplst_ );
        }
        virtual void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "image_list" ) )    path_imglst_ = val;
            if( !strcmp( name, "image_root") )     path_imgdir_ = val;
            if( !strcmp( name, "grey_scale" ) )    grey_scale_ = atoi( val );
            if( !strcmp( name, "silent"   ) )      silent_ = atoi( val );
        }
        virtual void Init( void ){
            fplst_  = utils::FopenCheck( path_imglst_.c_str(), "r" );
            if( silent_ == 0 ){
                printf("ImageIterator:image_list=%s, grey=%d\n", path_imglst_.c_str(), grey_scale_ );
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
                    this->LoadImage( fname );
                }else{
                    sprintf( sname, "%s%s", path_imgdir_.c_str(), fname );
                    this->LoadImage( sname );
                }
                return true;
            }
            return false;
        }
        virtual const DataInst &Value( void ) const{
            return out_;
        }
    private:
        inline void LoadImage( const char *fname ){
            using namespace cimg_library;
            CImg<real_t> img( fname ), res;
            // todo add depth to channel
            utils::Assert( img.depth() == 1, "can not handle 3D image so far" );

            if( grey_scale_ ){
                res = img.norm( 2 );
                res.normalize( 0, 1 );
            }else{
                res = img * (1.0f/256.0f);
            }

            img_.Resize( mshadow::Shape3( res.spectrum(), res.height(), res.width() ) );
            for( index_t z = 0; z < img_.shape[2]; ++z ){
                for( index_t y = 0; y < img_.shape[1]; ++y ){
                    for( index_t x = 0; x < img_.shape[0]; ++x ){
                        img_[z][y][x] = res( x, y, z );
                    }
                }
            }
            out_.data = img_;
        }
    private:
        // silent
        int silent_;
        // whether enforce grey scale
        int grey_scale_;
        // output data
        DataInst out_;
        // file pointer to list file, information file
        FILE *fplst_;
        // prefix path of image folder, path to input lst, format: imageid label path
        std::string path_imgdir_, path_imglst_;
        // temp storage for image
        mshadow::TensorContainer<cpu,3> img_;
    };
};
#endif

