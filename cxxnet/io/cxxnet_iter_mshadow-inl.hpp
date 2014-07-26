#ifndef CXXNET_MSHADOW_ITER_INL_HPP
#define CXXNET_MSHADOW_ITER_INL_HPP
#pragma once

#include "mshadow/tensor_container.h"
#include "cxxnet_data.h"
#include "../utils/cxxnet_io_utils.h"
#include "../utils/cxxnet_global_random.h"

namespace cxxnet {
    // load from mshadow binary data
    class MShadowIterator: public IIterator<DataInst> {
    public:
        MShadowIterator( void ){
            img_.dptr = NULL;
            labels_.dptr = NULL;
            mode_ = 1;
            inst_offset_ = 0;
            silent_ = 0;
            shuffle_ = 0;
        }
        virtual ~MShadowIterator( void ){
            if( img_.dptr == NULL ){
                mshadow::FreeSpace( img_ );
                mshadow::FreeSpace( labels_ );
            }
        }
        virtual void SetParam( const char *name, const char *val ) {
            if( !strcmp( name, "silent") )       silent_ = atoi( val );            
            if( !strcmp( name, "shuffle") )      shuffle_ = atoi( val );
            if( !strcmp( name, "index_offset") ) inst_offset_ = atoi( val );
            if( !strcmp( name, "path_img") )     path_img_ = val;
            if( !strcmp( name, "path_label") )   path_label_ = val;            
        }
        // intialize iterator loads data in
        virtual void Init( void ) {
            this->LoadImage();
            this->LoadLabel();
            utils::Assert( img_.shape[3] == labels_.shape[0], "label and image much match each other" );
            if( shuffle_ ) this->Shuffle();
            if( silent_ == 0 ){
                mshadow::Shape<4> s = img_.shape;
                printf("MShadowTIterator: load %u images, shuffle=%d, data=%u,%u,%u,%u\n", 
                       (unsigned)img_.shape[2], shuffle_, s[3],s[2],s[1],s[0] );
            }
        }
        virtual void BeforeFirst( void ) {
            this->loc_ = 0;
        }
        virtual bool Next( void ) {
            if( loc_ < img_.shape[3] ){
                out_.label = labels_[loc_];
                out_.index = inst_[loc_];
                out_.data =  img_[loc_];
                ++ loc_; return true;
            } else{
                return false;
            }
        }
        virtual const DataInst &Value( void ) const {
            return out_;
        }
    private:
        inline void LoadImage( void ) {
            mshadow::utils::FileStream fs( utils::FopenCheck( path_img_.c_str(), "rb") );
            mshadow::LoadBinary( fs, img_, false );
            fs.Close();
        }
        inline void LoadLabel( void ) {
            mshadow::utils::FileStream fs( utils::FopenCheck( path_label_.c_str(), "rb") );
            mshadow::LoadBinary( fs, labels_, false );
            fs.Close();
            inst_.resize( labels_.shape[0] );
            for( size_t i = 0; i < inst_.size(); ++ i ){
                inst_[i] = (unsigned)i + inst_offset_;
            }
        }
        inline void Shuffle( void ){
            utils::Shuffle( inst_ );
            mshadow::TensorContainer<cpu,1> tmplabel( labels_.shape );
            mshadow::TensorContainer<cpu,4> tmpimg  ( img_.shape );
            for( size_t i = 0; i < inst_.size(); ++ i ){
                unsigned ridx = inst_[i] - inst_offset_;
                mshadow::Copy( tmpimg[i], img_[ridx] );
                tmplabel[i] = labels_[ ridx ];
            }
            // copy back
            mshadow::Copy( img_, tmpimg );
            mshadow::Copy( labels_, tmplabel );
        }
    private:
        // silent
        int silent_;
        // path
        std::string path_img_, path_label_;
        // output 
        DataInst out_;
        // whether do shuffle
        int shuffle_;
        // data mode
        int mode_;
        // current location
        index_t loc_;
        // image content 
        mshadow::Tensor<cpu,4> img_;
        // label content
        mshadow::Tensor<cpu,1> labels_;
        // instance index offset
        unsigned inst_offset_;
        // instance index
        std::vector<unsigned> inst_; 
    }; //class MShadowIterator
}; // namespace cxxnet
#endif // CXXNET_MSHADOW_ITER_INL_HPP
