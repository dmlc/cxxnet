#ifndef CXXNET_MNIST_ITER_INL_HPP
#define CXXNET_MNIST_ITER_INL_HPP
#pragma once

#include "../cxxnet_data.h"
#include "../utils/cxxnet_io.h"

namespace cxxnet {
    class MNISTIterator: public IIterator<DataBatch> {
    public:
        MNISTIterator( void ){
            img_.dptr = NULL;
            mode_ = 0;
        }
        virtual ~MNISTIterator( void ){
            if( img_.dptr != NULL ) delete []img_.dptr;
        }
        virtual void SetParam( const char *name, const char *val ) {
            if( !strcmp( name, "batch_size") )  batch_size_ = (index_t)atoi( val ); 
            if( !strcmp( name, "mode") )        mode_ = atoi( val );
            if( !strcmp( name, "shuffle") )     shuffle_ = atoi( val );
            if( !strcmp( name, "path_img") )    path_img = val;
            if( !strcmp( name, "path_label") )  path_label = val;            
        }
        // intialize iterator loads data in
        virtual void Init( void ) {
            this->LoadImage();
            this->LoadLabel();
            if( mode_ == 0 ){
                out_.data.shape = mshadow::Shape4( batch_size_, 1, 1, img_.shape[1] * img_.shape[0] );
            }else{
                out_.data.shape = mshadow::Shape4( batch_size_, 1, img_.shape[1], img_.shape[0] );
            }
            out_.inst_index = NULL;
            out_.data.shape.stride_ = out_.data.shape[0];
            if( shuffle_ ) this->Shuffle();
        }
        virtual void BeforeFirst( void ) {
            this->loc_ = 0;
        }
        virtual bool Next( void ) {
            if( loc_ + batch_size_ <= img_.shape[2] ){
                loc_ += batch_size_;
                out_.data.dptr = img_[ loc_ ].dptr;
                out_.labels = &label_[ loc_ ];
                return true;
            } else{
                return false;
            }
        }
        virtual const DataBatch &Value( void ) const {
            return out_;
        }
    private:
        inline void LoadImage( void ) {
            utils::GzFile gzimg( path_img.c_str(), "rb" );
            gzimg.ReadInt();
            int image_count = gzimg.ReadInt();
            int image_rows  = gzimg.ReadInt();
            int image_cols  = gzimg.ReadInt();
            img_.shape = mshadow::Shape3( image_count, image_rows, image_cols );
            img_.shape.stride_ = img_.shape[0];
            
            // allocate continuous memory
            img_.dptr = new float[ img_.shape.MSize() ];
            for (int i = 0; i < image_count; ++i) {
                for (int j = 0; j < image_rows; ++j) {
                    for (int k = 0; k < image_cols; ++k) {
                        img_[k][j][i] = gzimg.ReadByte();
                    }
                }
            }
            // normalize to 0-1
            img_ *= 1.0f / 256.0f;
        }        
        inline void LoadLabel( void ) {
            utils::GzFile gzlabel( path_label.c_str(), "rb" );
            gzlabel.ReadInt();
            int label_count = gzlabel.ReadInt();
            label_.resize( label_count );
            for( int i = 0; i < label_count; ++i ) {
                label_[i] = gzlabel.ReadByte();
            }
        }
        inline void Shuffle( void ){
            // TODO
        }
    private:
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
        std::vector<float> label_;
    }; //class MNISTIterator
}; // namespace cxxnet
#endif // CXXNET_MNIST_HPP
