#ifndef CXXNET_CIFAR_ITER_INL_HPP
#define CXXNET_CIFAR_ITER_INL_HPP
#pragma once

#include "mshadow/tensor_container.h"
#include "cxxnet_data.h"
#include "../utils/cxxnet_io_utils.h"
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
            loc_ = 0;
        }
        virtual ~CIFARIterator( void ){
            if( img_.dptr != NULL ) delete []img_.dptr;
        }
        virtual void SetParam( const char *name, const char *val ) {
            if( !strcmp( name, "silent") )       silent_ = atoi( val );
            if( !strcmp( name, "batch_size") )   batch_size_ = (index_t)atoi( val );
            if( !strcmp( name, "input_flat") )         mode_ = atoi( val );
            if( !strcmp( name, "shuffle") )      shuffle_ = atoi( val );
            if( !strcmp( name, "index_offset") ) inst_offset_ = atoi( val );
            if( !strcmp( name, "path") )     path_ = val;
            if( !strcmp( name, "test") && atoi(val) == 1) file_list_.push_back("test_batch.bin");
            if( !strcmp( name, "batch1") && atoi(val) == 1) file_list_.push_back("data_batch_1.bin");
            if( !strcmp( name, "batch2") && atoi(val) == 1) file_list_.push_back("data_batch_2.bin");
            if( !strcmp( name, "batch3") && atoi(val) == 1) file_list_.push_back("data_batch_3.bin");
            if( !strcmp( name, "batch4") && atoi(val) == 1) file_list_.push_back("data_batch_4.bin");
            if( !strcmp( name, "batch5") && atoi(val) == 1) file_list_.push_back("data_batch_5.bin");
        }
        // intialize iterator loads data in
        virtual void Init( void ) {
            this->Load();
            if( mode_ == 1 ){
                out_.data.shape = mshadow::Shape4(1,1,batch_size_,img_.shape[1] * img_.shape[0] * 3 );
            }else{
                out_.data.shape = mshadow::Shape4( batch_size_, 3, img_.shape[1], img_.shape[0] );
            }
            out_.inst_index = NULL;
            out_.data.shape.stride_ = out_.data.shape[0];
            out_.batch_size = batch_size_;
            if( shuffle_ ) this->Shuffle();
            if( silent_ == 0 ){
                mshadow::Shape<4> s = out_.data.shape;
                printf("CIFARIterator: load %u images, shuffle=%d, shape=%u,%u,%u,%u\n",
                       (unsigned)img_.shape[3], shuffle_, s[3],s[2],s[1],s[0] );
            }
        }
        virtual void BeforeFirst( void ) {
            this->loc_ = 0;
        }
        virtual bool Next( void ) {
            // Different Mode should use differnt boundary ?
            if( loc_ + batch_size_ <= img_.shape[3] ) {
                out_.data.dptr = img_[ loc_ ].dptr;
                out_.labels = &labels_[ loc_ ];
                out_.inst_index = &inst_[ loc_ ];
                loc_ += batch_size_;
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
            index_t image_count = 10000;
            index_t total_count = image_count * file_list_.size();
            labels_.resize(total_count);
            img_.shape = mshadow::Shape4( total_count, 3, 32, 32);
            img_.shape.stride_ = img_.shape[0];
            // allocate continuous memory
            img_.dptr = new float[ img_.shape.MSize() ];
            for (index_t cnt = 0; cnt < file_list_.size(); ++cnt) {
                std::string f_path = path_ + file_list_[cnt];                
                utils::StdFile file( f_path.c_str(), "rb" );
                for (index_t i = 0 ; i < image_count; ++i) {
                    labels_[i + cnt * image_count] = file.ReadByte();
                    inst_.push_back( (unsigned)(i + cnt * image_count) + inst_offset_ );
                    for (index_t c = 0; c < 3; ++c) {
                        for (index_t row = 0; row < 32; ++row) {
                            for (index_t col = 0; col < 32; ++col) {
                                img_[i + cnt * image_count][c][row][col] = file.ReadByte();
                            }
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
        std::string path_;
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
        mshadow::Tensor<cpu,4> img_;
        // label content
        std::vector<float> labels_;
        // instance index offset
        unsigned inst_offset_;
        // instance index
        std::vector<unsigned> inst_;
        // file_list
        std::vector<std::string> file_list_;
    }; //class CIFARIterator
}; // namespace cxxnet
#endif // CXXNET_CIFAR_ITER_INL_HPP
