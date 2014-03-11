#ifndef CXXNET_CIFAR_ITER_INL_HPP
#define CXXNET_CIFAR_ITER_INL_HPP
#pragma once

#include "mshadow/tensor_container.h"
#include "../cxxnet_data.h"
#include "../utils/cxxnet_io.h"
#include "../utils/cxxnet_global_random.h"

namespace cxxnet {
    class CIFARIterator: public IIterator<DataBatch> {
    public:
        CIFARIterator( void ){
            img_.dptr = NULL;
            mode_ = 0;
            inst_offset_ = 0;
            silent_ = 0;
            shuffle_ = 0;
        }
        virtual ~CIFARIterator( void ){
            if( img_.dptr != NULL ) delete []img_.dptr;
        }
        virtual void SetParam( const char *name, const char *val ) {
            if( !strcmp( name, "silent") )       silent_ = atoi( val );
            if( !strcmp( name, "batch_size") )   batch_size_ = (index_t)atoi( val );
            if( !strcmp( name, "mode") )         mode_ = atoi( val );
            if( !strcmp( name, "shuffle") )      shuffle_ = atoi( val );
            if( !strcmp( name, "index_offset") ) inst_offset_ = atoi( val );
            if( !strcmp( name, "path_img") )     path_img = val;
            if( !strcmp( name, "path_label") )   path_label = val;
        }
        // intialize iterator loads data in
        virtual void Init( void ) {
            this->Load();
            if( mode_ == 0 ){
                out_.data.shape = mshadow::Shape4(1,1,batch_size_,img_.shape[1] * img_.shape[0] * 3 );
            }else{
                out_.data.shape = mshadow::Shape4( batch_size_, 3, img_.shape[1], img_.shape[0] );
            }
            out_.inst_index = NULL;
            out_.data.shape.stride_ = out_.data.shape[0];
            if( shuffle_ ) this->Shuffle();
            if( silent_ == 0 ){
                mshadow::Shape<4> s = out_.data.shape;
                printf("CIFARIterator: load %u images, shuffle=%d, shape=%u,%u,%u,%u\n",
                       (unsigned)img_.shape[2], shuffle_, s[3],s[2],s[1],s[0] );
            }
        }
        virtual void BeforeFirst( void ) {
            this->loc_ = 0;
        }
        virtual bool Next( void ) {
            if( loc_ + batch_size_ < img_.shape[2] ){
                loc_ += batch_size_;
                out_.data.dptr = img_[ loc_ ].dptr;
                out_.labels = &labels_[ loc_ ];
                out_.inst_index = &inst_[ loc_ ];
                return true;
            } else{
                return false;
            }
        }
        virtual const DataBatch &Value( void ) const {
            return out_;
        }
    private:
        inline void Load( void ) {
            utils::BinFile bfile( path_img.c_str(), "rb" );
            int image_count = bfile.Size() / 3073;
            label_.resize(image_count);
            img_.shape = mshadow::Shape4( image_count, 3, 32, 32);
            img_.shape.stride_ = img_.shape[0];
            // allocate continuous memory
            img_.dptr = new float[ img_.shape.MSize() ];
            for (int i = 0; i < image_count; ++i) {
                int label = (int) bfile.ReadByte();
                label_.push_back(label);
                for (int c = 0; c < 3; ++c) {
                    for (int row = 0; row < 32; ++row) {
                        for (int col = 0; col < 32; ++col) {
                            img_[i][c][row][col] = (int)bfile.ReadByte();
                        }
                    }
                }
            }
            // normalize to 0-1
            img_ *= 1.0f / 256.0f;
        }
        inline void Shuffle( void ){
            utils::Shuffle( inst_ );
            std::vector<float> tmplabel( labels_.size() );
            mshadow::TensorContainer<cpu,4> tmpimg( img_.shape );
            for( size_t i = 0; i < inst_.size(); ++ i ){
                unsigned ridx = inst_[i] - inst_offset_;
                mshadow::Copy( tmpimg[i], img_[ridx] );
                tmplabel[i] = labels_[ ridx ];
            }
            // copy back
            mshadow::Copy( img_, tmpimg );
            labels_ = tmplabel;
        }
    private:
        // silent
        int silent_;
        // path
        std::string path_img, path_label;
        // output
        DataBatch out_;
        // whether do shuffle
        int shuffle_;
        // data mode
        int mode_;
        // current location
        index_t loc_;
        // batch size
        index_t batch_size_;
        // image content
        mshadow::Tensor<cpu,3> img_;
        // label content
        std::vector<float> labels_;
        // instance index offset
        unsigned inst_offset_;
        // instance index
        std::vector<unsigned> inst_;
    }; //class CIFARIterator
}; // namespace cxxnet
#endif // CXXNET_CIFAR_ITER_INL_HPP
