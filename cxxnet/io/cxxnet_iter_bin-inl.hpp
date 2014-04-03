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
            buffer_size_ = 128;
            path_imglst_ = "img.lst";
            path_imgbin_ = "img.bin";
        }
        virtual ~BinaryIterator( void ){
            if( fplst_ != NULL ){
                fclose( fplst_ ); delete fpbin_;
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
            if( !strcmp( path_imgbin_.c_str() + path_imgbin_.length() - 3, ".gz" ) ){
                fpbin_  = new utils::GzFile( path_imgbin_.c_str(), "rb" );
            }else{
                fpbin_  = new utils::StdFile( path_imgbin_.c_str(), "rb" );
            }
            
            utils::Assert( fpbin_->Read( &dshape_[3], sizeof(mshadow::index_t) ), "BinaryIterator: load header");
            utils::Assert( fpbin_->Read( &dshape_[2], sizeof(mshadow::index_t) ), "BinaryIterator: load header");
            utils::Assert( fpbin_->Read( &dshape_[1], sizeof(mshadow::index_t) ), "BinaryIterator: load header");
            utils::Assert( fpbin_->Read( &dshape_[0], sizeof(mshadow::index_t) ), "BinaryIterator: load header");

            img_.set_pad( false ); img_.Resize( dshape_.SubShape() );
            utils::Assert( img_.shape.Size() == img_.shape.MSize(), "BUG" );
            if( silent_ == 0 ){
                printf("BinaryIterator:image_list=%s, image_bin=%s, shape=%u,%u,%u,%u\n", 
                       path_imglst_.c_str(), path_imgbin_.c_str(), dshape_[3], dshape_[2], dshape_[1], dshape_[0] );
            }
            buf_.resize( buffer_size_ * img_.shape.Size() );
            this->BeforeFirst();
        }
        virtual void BeforeFirst( void ){
            fpbin_->Seek( sizeof(mshadow::index_t) * 4 );
            fseek( fplst_, 0, SEEK_SET );
            num_readed_ = 0; buf_top_ = 0; num_inbuffer_ = 0;
        }
        virtual bool Next( void ){
            while( fscanf( fplst_,"%u\t%f%*[^\n]\n", &out_.index, &out_.label ) == 2 ){
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
            if( buf_top_ >= num_inbuffer_ ){
                if( num_inbuffer_ == 0 ){
                    num_inbuffer_ = std::min( dshape_[3] - num_readed_, buffer_size_ );
                    utils::Assert( num_inbuffer_ > 0 && num_readed_ <= dshape_[3], "list longer than binary file");
                    utils::Assert( fpbin_->Read( &buf_[0], sizeof(unsigned char) * num_inbuffer_ * img_.shape.Size()), "read buffer" );
                }
                buf_top_ = 0; num_readed_ += num_inbuffer_;
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
        // number of instances in buffer
        size_t num_inbuffer_;
        // number of instances readed
        size_t num_readed_;
        // buffer size
        size_t buffer_size_;
        // buffer
        std::vector<unsigned char> buf_;
        // shape of data
        mshadow::Shape<4> dshape_;
        // file pointer to list file, information file
        FILE *fplst_;
        // pointer to binary file 
        utils::ISeekStream *fpbin_;
        // prefix path of image binary file, path to input lst, format: imageid label path
        std::string path_imgbin_, path_imglst_;
        // temp storage for image
        mshadow::TensorContainer<cpu,3> img_;
    };
};
#endif

