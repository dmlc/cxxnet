#ifndef CXXNET_PAIRTEST_INL_HPP
#define CXXNET_PAIRTEST_INL_HPP
#pragma once
/*!
 * \file cxxnet_pairtest-inl.hpp
 * \brief module to do pairtest, used to compare layer implementations
 * \author Tianqi Chen, Bing Xu
 */
#include "cxxnet_core.h"

namespace cxxnet{
    template<typename xpu>
    struct PairTestLayer : public ILayer{
    protected:
        class PairTestUpdater: public IUpdater{
        public:
            PairTestUpdater( IUpdater *umaster, IUpdater *uslave, const char *tag )
                :umaster_(umaster), uslave_(uslave), tag_(tag){
                w_mst_.set_pad( false ); w_slv_.set_pad( false );
                g_mst_.set_pad( false ); g_slv_.set_pad( false );
                sync_weight_  = 1;
            }
            inline void Sync( void ){
                umaster_->GetData( w_mst_, g_mst_ );
                uslave_->SetData( w_mst_, g_mst_ );
            }
            virtual void Init( void ){
                umaster_->Init(); uslave_->Init();
            }
            virtual void Update( long epoch ){
                umaster_->Update( epoch );  uslave_->Update( epoch );
                umaster_->GetData( w_mst_, g_mst_ );
                uslave_->GetData( w_slv_, g_slv_ );
                CmpResult( w_mst_, w_slv_, "update:weight", tag_ );
                CmpResult( g_mst_, g_slv_, "update:gradient", tag_ );
                if( sync_weight_ != 0 ) this->Sync();
            }
            virtual void StartRound( int round ) {
                umaster_->StartRound( round );
                uslave_->StartRound( round );
            }
            virtual void SetParam( const char *name, const char *val ) {
                umaster_->SetParam( name, val );
                uslave_->SetParam( name, val );
                if( !strcmp( name, "sync_weight") ) sync_weight_ = atoi(val);
            }
            virtual void SetData(const mshadow::Tensor<cpu,2>& weight,
                                 const mshadow::Tensor<cpu,2>& gradient) {
            }
            virtual void GetData(mshadow::TensorContainer<cpu,2>& weight,
                                 mshadow::TensorContainer<cpu,2>& gradient ) const {
            }
        private:
            int sync_weight_;
            IUpdater *umaster_, *uslave_;
            const char *tag_;
            mshadow::TensorContainer<cpu,2> w_mst_, w_slv_;
            mshadow::TensorContainer<cpu,2> g_mst_, g_slv_;
        };
    public:
        PairTestLayer( mshadow::Random<xpu> &rnd, Node<xpu>&in, Node<xpu>& out,
                       int tmaster, int tslave ):in_(in),out_(out){
            master_ = CreateLayer_( tmaster, rnd, in, out ) ;
            slave_  = CreateLayer_( tslave, rnd, slv_in_, slv_out_ );
        }
        virtual ~PairTestLayer( void ){
            delete master_; delete slave_;
            slv_in_.FreeSpace(); slv_out_.FreeSpace();
        }
        virtual void Forward( bool is_train ){
            Copy( slv_in_.data, in_.data );
            master_->Forward( is_train );
            slave_->Forward( is_train );
            CmpResult( out_.data.FlatTo2D(), slv_out_.data.FlatTo2D(), "forward" );
        }
        virtual void Backprop( bool prop_grad ){
            Copy( slv_out_.data, out_.data );
            master_->Backprop( prop_grad );
            slave_->Backprop( prop_grad );
            if( prop_grad ){
                CmpResult( in_.data.FlatTo2D(), slv_in_.data.FlatTo2D(), "backprop" );
            }
        }
    public:
        virtual void InitLayer( void ){
            slv_in_.data.shape = in_.data.shape;
            master_->InitLayer();
            slave_->InitLayer();
            utils::Assert( slv_out_.data.shape == out_.data.shape, "PairTestLayer:InitLayer mismatch" );
            AllocSpace( slv_in_.data );
            AllocSpace( slv_out_.data );
        }
        virtual void GetUpdaters( const char *updater, std::vector<IUpdater*> &updaters ) {
            std::vector<IUpdater*> umaster, uslave;
            master_->GetUpdaters( updater, umaster );
            slave_->GetUpdaters( updater, uslave );
            utils::Assert( umaster.size() == uslave.size(), "PairTestLayer: number of updaters not match" );
            for( size_t i = 0; i < umaster.size(); ++i ){
                PairTestUpdater *up = new PairTestUpdater( umaster[i], uslave[i], i==0?"-wmat":"-bias" );
                up->Sync(); updaters.push_back( up );
            }
        }
        virtual void SetParam( const char *name, const char* val ) {
            master_->SetParam( name, val );
            slave_->SetParam( name, val );
            if( !strncmp( name, "master:", 7 ) ) master_->SetParam( name+7, val );
            if( !strncmp( name, "slave:", 6 ) ) slave_->SetParam( name+7, val );
        }
        virtual void InitModel(void) {
            master_->InitModel();
            slave_->InitModel();
        }
        virtual void SaveModel(mshadow::utils::IStream &fo) const {
            master_->SaveModel( fo );
            slave_->SaveModel( fo );
        }
        virtual void LoadModel(mshadow::utils::IStream &fi) {
            master_->LoadModel( fi );
            slave_->LoadModel( fi );
        }
    private:
        template<typename xxpu>
        inline static void CmpResult( mshadow::Tensor<xxpu,2> dmaster, mshadow::Tensor<xxpu,2> dslave,
                                      const char *tag, const char *tag2="" ){
            mshadow::TensorContainer<cpu, 2> tmst(false), tslv(false);
            tmst.Resize( dmaster.shape ); tslv.Resize( dslave.shape );
            mshadow::Copy( tmst, dmaster ); mshadow::Copy( tslv, dslave );
            index_t count = tmst.shape.Size();
            double diff = 0.0, ssum = 0.0, maxdiff = 0.0;
            index_t mxidx = 0;
            for( index_t i = 0; i < count; ++i ){
                double d = std::abs( tmst.dptr[i] - tslv.dptr[i] );
                if( d > maxdiff ){
                    maxdiff = d; mxidx = i;
                }
                diff += d;
                ssum += std::abs( tmst.dptr[i] );
            }
            // relative absolute error
            double rerr = diff / ssum;
            if( rerr > 1e-5 || diff != diff ){
                fprintf( stderr, "%s%s: err=%f, maxd[%u]=%f, diff=%f, ssum=%f\n", tag, tag2, rerr, mxidx, maxdiff, diff, ssum );
            }
        }
    private:
        ILayer *master_, *slave_;
        Node<xpu> &in_, &out_;
        // data that slave takes
        Node<xpu> slv_in_, slv_out_;
    };
};

#endif

