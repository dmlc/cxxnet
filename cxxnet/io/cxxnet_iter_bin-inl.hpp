#ifndef CXXNET_ITER_BINARY_INL_HPP
#define CXXNET_ITER_BINARY_INL_HPP
#pragma once
/*!
 * \file cxxnet_iter_bin-inl.hpp
 * \brief implementation of binary iterator
 * \author Tianqi Chen
 */
#include "mshadow/tensor_container.h"
#include "../utils/cxxnet_io_utils.h"
#include "cxxnet_data.h"

namespace cxxnet{
    /*! \brief simple binary iterator that reads data from binary format */
    class BinaryIterator : public IIterator< DataInst >{
    public:
        BinaryIterator( void ){
            fpbin_ = NULL;
            fplst_ = NULL;
            silent_ = 0;
            buffer_size_ = 1000;
            path_imglst_ = "img.lst";
            path_imgbin_ = "img.bin";
        }
        virtual ~BinaryIterator( void ){
            if( fplst_ != NULL ){
                fclose( fpbin_ ); fclose( fplst_ );
            }
        }
        virtual void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "image_list" ) )    path_imglst_ = val;
            if( !strcmp( name, "image_bin") )      path_imgbin_ = val;
            if( !strcmp( name, "image_bin_buffer") ) buffer_size_ = (size_t)atoi(val);
            if( !strcmp( name, "silent"   ) )        silent_ = atoi( val );
        }
        virtual void Init( void ){
            fplst_  = utils::FopenCheck( path_imglst_.c_str(), "r" );
            fpbin_  = utils::FopenCheck( path_imgbin_.c_str(), "rb" );

            utils::Assert( fread( &dshape_, sizeof(dshape_), 1, fpbin_ ), "BinaryIterator: load header");
            img_.set_pad( false ); img_.Resize( dshape_.SubShape() );
            utils::Assert( img_.shape.Size() == img_.shape.MSize(), "BUG" );

            if( silent_ == 0 ){
                printf("BinaryIterator:image_list=%s, image_bin=%s\n", path_imglst_.c_str(), path_imgbin_.c_str() );
            }
            this->BeforeFirst();

        }
        virtual void BeforeFirst( void ){
            fseek( fpbin_ , sizeof(dshape_), SEEK_SET );
            buf_.reserve( buffer_size_ ); buf_.clear(); 
            num_readed_ = 0; buf_top_ = 0;
        }
        virtual bool Next( void ){
            while( fscanf( fplst_,"%u%f%*[^\n]\n", &out_.index, &out_.label ) == 3 ){
                this->LoadImage();
                return true;
            }
            return false;
        }
        virtual const DataInst &Value( void ) const{
            return out_;
        }
    private:
        inline void LoadImage( void ){
            if( buf_top_ >= buf_.size() ){
                buf_.resize( std::min( dshape_[3] - num_readed_, buffer_size_ ) );
                utils::Assert( buf_.size() != 0, "binary file data is smaller than listed in list" );
                fread( &buf_[0], sizeof(unsigned char), buffer_size_ * img_.shape.Size(), fpbin_ );
                num_readed_ += buf_.size();
            }
            mshadow::index_t n = img_.shape.Size();
            for( mshadow::index_t i = 0; i < n; ++ i ){
                img_.dptr[i] = static_cast<mshadow::real_t>( buf_[ buf_top_*n + i ] );
            }
            out_.data = img_; buf_top_ += 1;
        }
    private:
        // silent
        int silent_;
        // output data
        DataInst out_;                
        // number of instances readed in buffer
        size_t buf_top_;
        // number of instances readed
        size_t num_readed_;
        // buffer size
        size_t buffer_size_;
        // buffer
        std::vector<unsigned char> buf_;
        // shape of data
        mshadow::Shape<4> dshape_;      
        // file pointer to list file, information file
        FILE *fpbin_, *fplst_;
        // prefix path of image binary file, path to input lst, format: imageid label path
        std::string path_imgbin_, path_imglst_;
        // temp storage for image
        mshadow::TensorContainer<cpu,3> img_;
    };
};
#endif

