#ifndef CXXNET_CAFFE_ADAPTER_INL_HPP
#define CXXNET_CAFEE_ADAPTER_INL_HPP
#pragma once
/*!
 * \file cxxnet_caffee_adapter-inl.hpp
 * \brief try to adapt caffe layers, this code comes as plugin of cxxnet, and by default not included in the code.
 * \author Tianqi Chen
 */
#include <climits>
#include "caffe/caffe.hpp"
#include "mshadow/tensor.h"
#include "mshadow/tensor_container.h"
#include <google/protobuf/text_format.h>

namespace cxxnet{
    using namespace mshadow::expr;
    using namespace mshadow::utils;
    /*!
     * \brief adapter from caffe, will cost a extra blob memory,
     *        but allows some correct comparisons
     */
    template<typename xpu>
    class CaffeLayer: public ILayer{
    private:
        class CaffeUpdater: public IUpdater{
        public:
            CaffeUpdater( const char *updater, mshadow::Random<xpu> &rnd,
                          caffe::Blob<real_t>* blb, const char *tag, int batch_size )
                :blb_(blb), batch_size_(batch_size){
                this->tag = tag;
                if( xpu::kDevCPU ){
                    weight_.dptr = blb->mutable_cpu_data();
                    grad_.dptr = blb->mutable_cpu_diff();
                }else{
                    weight_.dptr = blb->mutable_gpu_data();
                    grad_.dptr = blb->mutable_gpu_diff();
                }
                weight_.shape[0] = weight_.shape.stride_ = blb->count();
                grad_.shape = weight_.shape;
                base_ = CreateUpdater<xpu>( updater, rnd, weight_, grad_, tag );
            }
            virtual ~CaffeUpdater( void ){
                delete base_;
            }
            virtual void Init( void ){
                base_->Init();
            }
            virtual void Update( long epoch ) {
                if( xpu::kDevCPU ){
                    utils::Assert( blb_->mutable_cpu_data() == weight_.dptr, "CaffeUpdater" );
                    utils::Assert( blb_->mutable_cpu_diff() == grad_.dptr, "CaffeUpdater" );
                }else{
                    utils::Assert( blb_->mutable_gpu_data() == weight_.dptr, "CaffeUpdater" );
                    utils::Assert( blb_->mutable_gpu_diff() == grad_.dptr, "CaffeUpdater" );
                }
                base_->Update( epoch );
            }
            virtual void StartRound( int round ) {
                base_->StartRound( round );
            }
            virtual void SetParam( const char *name, const char *val ){
                if( !strncmp( name, tag.c_str(), tag.length() ) ){
                    if( name[tag.length()] == ':' ) name += tag.length() + 1;
                }
                base_->SetParam( name, val );
            }
            virtual void SetData(const mshadow::Tensor<cpu,2>& weight,
                                 const mshadow::Tensor<cpu,2>& gradient) {
                base_->SetData(weight, gradient);
            }
            virtual void GetData(mshadow::TensorContainer<cpu,2>& weight,
                                 mshadow::TensorContainer<cpu,2>& gradient ) const {
                base_->GetData(weight, gradient);
            }
        private:
            std::string tag;
            IUpdater *base_;
            caffe::Blob<real_t> * blb_;
            int batch_size_;
            mshadow::Tensor<xpu,1> weight_, grad_;
        };
    public:
        CaffeLayer( mshadow::Random<xpu> &rnd, Node<xpu>& in, Node<xpu>& out )
            :rnd_(rnd), in_(in), out_(out){
            this->base_ = NULL;
            this->mode_ = -1;
            this->blb_in_ = NULL;
            this->blb_out_ = NULL;
        }
        virtual ~CaffeLayer( void ){
            this->FreeSpace();
            if( blb_in_ != NULL )  delete blb_in_;
            if( blb_out_ != NULL ) delete blb_out_;
        }
        virtual void Forward( bool is_train ){
            mshadow::Shape<4> shape_in = in_.data.shape; shape_in.stride_ = shape_in[0];
            mshadow::Shape<4> shape_ou = out_.data.shape; shape_ou.stride_ = shape_ou[0];
            if( xpu::kDevCPU ){
                mshadow::Tensor<xpu,4> tbin( blb_in_->mutable_cpu_data(), shape_in );
                mshadow::Copy( tbin, in_.data );
                base_->Forward( vec_in_, &vec_out_ );
                mshadow::Tensor<xpu,4> tbout( blb_out_->mutable_cpu_data(), shape_ou );
                mshadow::Copy( out_.data, tbout );
            }else{
                mshadow::Tensor<xpu,4> tbin( blb_in_->mutable_gpu_data(), shape_in );
                mshadow::Copy( tbin, in_.data );
                base_->Forward( vec_in_, &vec_out_ );
                mshadow::Tensor<xpu,4> tbout( blb_out_->mutable_gpu_data(), shape_ou );
                mshadow::Copy( out_.data, tbout );
            }
        }
        virtual void Backprop( bool prop_grad ){
            mshadow::Shape<4> shape_in = in_.data.shape; shape_in.stride_ = shape_in[0];
            mshadow::Shape<4> shape_ou = out_.data.shape; shape_ou.stride_ = shape_ou[0];
            if( xpu::kDevCPU ){
                mshadow::Tensor<xpu,4> tbout( blb_out_->mutable_cpu_diff(), shape_ou );
                mshadow::Copy( tbout, out_.data );
                base_->Backward( vec_out_, prop_grad, &vec_in_ );
                if( prop_grad ){
                    mshadow::Tensor<xpu,4> tbin( blb_in_->mutable_cpu_diff(), shape_in );
                    mshadow::Copy( in_.data, tbin );
                }
            }else{
                mshadow::Tensor<xpu,4> tbout( blb_out_->mutable_gpu_diff(), shape_ou );
                mshadow::Copy( tbout, out_.data );
                base_->Backward( vec_out_, prop_grad, &vec_in_ );
                if( prop_grad ){
                    mshadow::Tensor<xpu,4> tbin( blb_in_->mutable_gpu_diff(), shape_in );
                    mshadow::Copy( in_.data, tbin );
                }
            }
        }
        virtual void InitLayer( void ){
            utils::Assert( mode_ != -1, "CaffeLayer: must specify mode: 0:flatten, 1:conv-channels" );
            mshadow::Shape<4> ishape = in_.data.shape;
            if( mode_ == 0 ){
                utils::Assert( ishape[3] == 1 && ishape[2] == 1, "the input is not flattened, forget a FlattenLayer?" );
                batch_size_ = ishape[1];
                blb_in_  = new caffe::Blob<real_t>( ishape[1], ishape[0], 1, 1 );
                blb_out_ = new caffe::Blob<real_t>();
            }else{
                batch_size_ = in_.data.shape[3];
                blb_in_  = new caffe::Blob<real_t>( ishape[3], ishape[2], ishape[1], ishape[0] );
                blb_out_ = new caffe::Blob<real_t>();
            }
            vec_in_.clear(); vec_in_.push_back( blb_in_ );
            vec_out_.clear(); vec_out_.push_back( blb_out_ );

            if( base_ == NULL ){
                base_ = caffe::GetLayer<real_t>( param_ );
            }

            base_->SetUp( vec_in_, &vec_out_ );
            if( mode_ == 0 || mode_ == 2 ){
                out_.data.shape = mshadow::Shape4( 1, 1, blb_out_->num(), blb_out_->channels() );
            }else{
                out_.data.shape = mshadow::Shape4( blb_out_->num(), blb_out_->channels(), blb_out_->height(), blb_out_->width() );
            }
        }
        virtual void GetUpdaters( const char *updater, std::vector<IUpdater*> &updaters ) {
            const std::vector<boost::shared_ptr<caffe::Blob<real_t> > > &blobs = base_->blobs();
            for( size_t i = 0; i < blobs.size(); ++ i ){
                // Assume that blobs do not change
                char tag[ 256 ];
                sprintf( tag, "blob%d", (int)i );
                updaters.push_back( new CaffeUpdater( updater, rnd_, blobs[i].get(), tag, batch_size_ ) );
            }
        }
        virtual void SetParam( const char *name, const char* val ) {
            if( !strcmp( name, "proto") ){
                google::protobuf::TextFormat::ParseFromString( std::string(val), &param_ );
            }
            if( !strcmp( name, "mode" ) ){
                mode_ = atoi( val );
            }
            if( !strcmp( name, "dev" ) ){
                if( !strcmp( val, "cpu") ) caffe::Caffe::set_mode( caffe::Caffe::CPU );
                if( !strcmp( val, "gpu") ) caffe::Caffe::set_mode( caffe::Caffe::GPU );
            }
        }
        virtual void InitModel(void) {
        }
        virtual void SaveModel(mshadow::utils::IStream &fo) const {
            std::vector<char> buf;
            caffe::LayerParameter lparam = base_->layer_param();
            base_->ToProto( &lparam );            
            int msize = lparam.ByteSize();
            buf.resize( msize );
            fo.Write( &msize, sizeof(int) );
            utils::Assert( lparam.SerializeToArray( &buf[0], msize  ) );
            fo.Write( &buf[0], msize );
        }
        virtual void LoadModel(mshadow::utils::IStream &fi) {
            int msize;
            std::vector<char> buf;
            fi.Read( &msize, sizeof(int) );
            buf.resize( msize );
            utils::Assert( fi.Read( &buf[0], msize )!= 0, "CaffeLayer::LoadModel" );
            param_.ParseFromArray( &buf[0], msize );
            this->FreeSpace();
            base_ = caffe::GetLayer<real_t>( param_ );
        }
    private:
        inline void FreeSpace( void ){
            if( base_ != NULL ) delete base_;
            base_ = NULL;
        }
    private:
        /* !\brief random number generater */
        mshadow::Random<xpu>& rnd_;
        Node<xpu> &in_, &out_;
        /*!\brief mini batch size*/
        int batch_size_;
        /*!\brief whether it is fullc or convolutional layer */
        int mode_;
        /*! \brief caffe's layer parametes */
        caffe::LayerParameter param_;
        /*! \brief caffe's impelementation */
        caffe::Layer<real_t>* base_;
        /*! \brief blob data */
        caffe::Blob<real_t>* blb_in_;
        caffe::Blob<real_t>* blb_out_;
        /*!\ brief stores blb in */
        std::vector< caffe::Blob<real_t>* > vec_in_;
        std::vector< caffe::Blob<real_t>* > vec_out_;
    };
};

#endif
