#ifndef CXXNET_ITER_SPFEAT_INL_HPP
#define CXXNET_ITER_SPFEAT_INL_HPP
#pragma once

#include "mshadow/tensor_container.h"
#include "cxxnet_data.h"
#include "../utils/cxxnet_io_utils.h"

namespace cxxnet {
    class SpFeatIterator: public IIterator<DataBatch> {
    public:
        SpFeatIterator( void ){
            mode_ = 0;
            inst_offset_ = 0;
            silent_ = 0;
            shuffle_ = 0;
        }
        virtual ~SpFeatIterator( void ){
        }
        virtual void SetParam( const char *name, const char *val ) {
            if( !strcmp( name, "silent") )       silent_ = atoi( val );            
            if( !strcmp( name, "batch_size") )   batch_size_ = (index_t)atoi( val ); 
            if( !strcmp( name, "shuffle") )      shuffle_ = atoi( val );
            if( !strcmp( name, "index_offset") ) inst_offset_ = atoi( val );
            if( !strcmp( name, "path") )         path = val;
        }
        // intialize iterator loads data in
        virtual void Init( void ) {
            FILE *fi = utils::FopenCheck( path.c_str(), "r" );
            int ymax, xmax;
            utils::Assert( fscanf( fi, "%d%d", &ymax, &xmax ) == 2 );
            labels_.resize( ymax );

            img_.Resize( mshadow::Shape2( ymax, xmax ), 0.0f );
            for( index_t i = 0; i < img_.shape[1]; ++i ){
                int findex, cnt;
                float ylabel, fvalue; 
                utils::Assert( fscanf( fi, "%f%d", &ylabel, &cnt ) == 2 );
                labels_[i] = ylabel;
                utils::Assert( inst_.size() == i );
                inst_.push_back( (unsigned)i + inst_offset_ );
                while( cnt -- ){
                    utils::Assert( fscanf( fi, "%d:%f", &findex, &fvalue ) == 2 );
                    utils::Assert( findex < xmax, "feature index exceed bound" );
                    img_[i][findex] = fvalue;
                }
            }
            fclose( fi );

            out_.data.shape = mshadow::Shape4(1,1,batch_size_,img_.shape[0] );
            out_.inst_index = NULL;
            out_.data.shape.stride_ = img_.shape.stride_;
            out_.batch_size = batch_size_;
            if( shuffle_ ) this->Shuffle();
            if( silent_ == 0 ){
                mshadow::Shape<4> s = out_.data.shape;
                printf("SpfeatIterator: load %u data, shuffle=%d, shape=%u,%u,%u,%u\n", 
                       (unsigned)img_.shape[1], shuffle_, s[3],s[2],s[1],s[0] );
            }
        }
        virtual void BeforeFirst( void ) {
            this->loc_ = 0;
        }
        virtual bool Next( void ) {
            if( loc_ + batch_size_ <= img_.shape[1] ){
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
        inline void Shuffle( void ){
            utils::Shuffle( inst_ );
            std::vector<float> tmplabel( labels_.size() );
            mshadow::TensorContainer<cpu,2> tmpimg( img_.shape );
            for( size_t i = 0; i < inst_.size(); ++i ){
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
        std::string path;
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
        mshadow::TensorContainer<cpu,2> img_;
        // label content
        std::vector<float> labels_;
        // instance index offset
        unsigned inst_offset_;
        // instance index
        std::vector<unsigned> inst_; 
    }; //class SpfeatIterator
}; // namespace cxxnet

#endif // CXXNET_ITER_SPFEAT_INL_HPP
