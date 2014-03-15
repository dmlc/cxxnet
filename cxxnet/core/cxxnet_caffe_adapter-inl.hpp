#ifndef CXXNET_CAFFE_ADAPTER_INL_HPP
#define CXXNET_CAFEE_ADAPTER_INL_HPP
#pragma once
/*!
 * \file cxxnet_caffee_adapter-inl.hpp
 * \brief try to adapt caffe layers 
 * \author Tianqi Chen
 */
#include <climits>
#include "caffe/caffe.hpp"
#include "mshadow/tensor.h"
#include "mshadow/tensor_container.h"

namespace cxxnet{
    using namespace mshadow::expr;
    using namespace mshadow::utils;
    
    /*! 
     * \brief adapter from caffe, will cost a extra blob memory, 
     *        but allows some correct comparisons
     */
    template<typename xpu>
    class CaffeLayer: public ILayer{
    public:
        class CaffeUpdater : public IUpdater{
        public:
            CaffeUpdater( caffe::Layer<real_t> *base ):base_(base){}
            virtual void Init( void ) {
            }
            virtual void Update( void ) {
                std::vector<boost::shared_ptr<caffe::Blob<real_t> > >& blobs = base_->blobs();
                for( size_t i = 0; i < blobs.size(); ++ i ){
                    blobs[i]->Update();
                }
            }
            virtual void StartRound( int round ) {}
            virtual void SetParam( const char *name, const char *val ) {}
        private:
            // base class 
            caffe::Layer<real_t> *base_;            
        };
    public:
        CaffeLayer( Node<xpu>& in, Node<xpu>& out )
            :in_(in), out_(out){
            this->base_ = NULL;
            this->mode_ = -1;
            this->blb_in_ = NULL;
            this->blb_out_ = NULL;
            this->oshape_[0] = 0;
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
                mshadow::Tensor<xpu,4> tbout( blb_out_->mutable_cpu_data(), shape_ou );
                mshadow::Copy( tbin, in_.data );
                base_->Forward( vec_in_, &vec_out_ );
                mshadow::Copy( out_.data, tbout );
            }else{
                mshadow::Tensor<xpu,4> tbin( blb_in_->mutable_gpu_data(), shape_in );
                mshadow::Tensor<xpu,4> tbout( blb_out_->mutable_gpu_data(), shape_ou );
                mshadow::Copy( tbin, in_.data );
                base_->Forward( vec_in_, &vec_out_ );
                mshadow::Copy( out_.data, tbout );                
            }
        }
        virtual void Backprop( bool prop_grad ){
            mshadow::Shape<4> shape_in = in_.data.shape; shape_in.stride_ = shape_in[0];
            mshadow::Shape<4> shape_ou = out_.data.shape; shape_ou.stride_ = shape_ou[0];
            if( xpu::kDevCPU ){
                mshadow::Tensor<xpu,4> tbin( blb_in_->mutable_cpu_diff(), shape_in );
                mshadow::Tensor<xpu,4> tbout( blb_out_->mutable_cpu_diff(), shape_ou );
                mshadow::Copy( tbout, out_.data );
                base_->Backward( vec_out_, prop_grad, &vec_in_ );
                if( prop_grad ){
                    mshadow::Copy( in_.data, tbin );
                }
            }else{
                mshadow::Tensor<xpu,4> tbin( blb_in_->mutable_cpu_diff(), shape_in );
                mshadow::Tensor<xpu,4> tbout( blb_out_->mutable_cpu_diff(), shape_ou );
                mshadow::Copy( tbout, out_.data );
                base_->Backward( vec_out_, prop_grad, &vec_in_ );
                if( prop_grad ){
                    mshadow::Copy( in_.data, tbin );
                }
            }            
        }
        virtual void AdjustNodeShape( void ){ 
            utils::Assert( mode_ != -1, "CaffeLayer: must specify mode: 0:flatten, 1:conv-channels" );
            utils::Assert( oshape_[0] != 0, "CaffeLayer: must specify oshape" );
            if( mode_ == 0 ){
                out_.data.shape = mshadow::Shape4( 1, 1, in_.data.shape[1], oshape_[0] );
                blb_in_  = new caffe::Blob<real_t>( in_.data.shape[1], in_.data.shape[0], 1, 1 );
                blb_out_ = new caffe::Blob<real_t>( out_.data.shape[1], out_.data.shape[0], 1, 1 );
            }else{
                out_.data.shape = mshadow::Shape4( in_.data.shape[4], oshape_[2], oshape_[1], oshape_[0] );
                blb_in_  = new caffe::Blob<real_t>( in_.data.shape[3], in_.data.shape[2], in_.data.shape[1], in_.data.shape[0] );
                blb_out_ = new caffe::Blob<real_t>( out_.data.shape[3], out_.data.shape[2], out_.data.shape[1], out_.data.shape[0] );
            }
            vec_in_.clear(); vec_in_.push_back( blb_in_ );
            vec_out_.clear(); vec_out_.push_back( blb_out_ );            
            base_->SetUp( vec_in_, &vec_out_ ); 
        }   
        virtual void GetUpdaters( const char *updater, std::vector<IUpdater*> &updaters ) {
            updaters.push_back( new CaffeUpdater( base_ ) );
        }   
        virtual void SetParam( const char *name, const char* val ) {
            if( !strcmp( name, "proto") ){
                param_.ParseFromString( val );
            }
            if( !strcmp( name, "oshape") ){
                unsigned zmax, ymax, xmax;
                utils::Assert( sscanf( val, "%u,%u,%u", &zmax, &ymax, &xmax ) == 3, "CaffeLayer::SetParam" );
                oshape_ =  mshadow::Shape3( zmax, ymax, xmax );
            }
            if( !strcmp( name, "dev" ) ){
                if( !strcmp( val, "cpu") ) caffe::Caffe::set_mode( caffe::Caffe::CPU );
                if( !strcmp( val, "gpu") ) caffe::Caffe::set_mode( caffe::Caffe::GPU );
            }
        }
        virtual void InitModel(void) {
            this->FreeSpace();
            base_ = caffe::GetLayer<real_t>( param_ );            
        }
        virtual void SaveModel(mshadow::utils::IStream &fo) const {
            std::vector<char> buf;
            const caffe::LayerParameter& lparam = base_->layer_param();
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
        /*!\brief whether it is fullc or convolutional layer */
        int mode_;
        /*! \brief shape of output except batch size */
        mshadow::Shape<3> oshape_;
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
        Node<xpu> &in_, &out_;
    };
};

#endif
