#ifndef CXXNET_LAYER_INL_HPP
#define CXXNET_LAYER_INL_HPP
#pragma once
/*!
 * \file cxxnet_layer-inl.hpp
 * \brief implementation of different layers
 * \author Bing Xu, Tianqi Chen
 */

#include "cxxnet_net.h"
#include "mshadow/tensor_container.h"

namespace cxxnet{
    // expr is needed to use expression
    using namespace mshadow::expr;
    using namespace mshadow::utils;
    
    // simple fully connected layer that connects two nodes
    template<typename xpu>
    class FullConnectLayer{
    public:
        FullConnectLayer( Node<xpu> &in, Node<xpu> &out )
            :in_(in), out_(out){ 
            Assert( in_.data.shape[1] == out_.data.shape[1], "input output batch mismatch" );
            wmat_.Resize( mshadow::Shape2( in_.data.shape[0], out_.data.shape[0] ) );
            gwmat_.Resize( wmat_.shape );
            bias_.Resize( mshadow::Shape1( out_.data.shape[0] ) );
            gbias_.Resize( gbias_.shape );
        }            
        virtual ~FullConnectLayer( void ){            
        }
        virtual void Forwardprop(bool is_train) {
            index_t nbatch = in_.data.shape[1];
            out_.mat()  = dot( in_.mat(), wmat_ );
            out_.mat() += repmat( bias_, nbatch );
        }
        virtual void Backprop(bool is_firstlayer){
            index_t nbatch = in_.data.shape[1];
            real_t scale = 1.0f / nbatch; 
            gwmat_ = scale * dot( in_.mat().T(), out_.mat() );
            // todo sum
            // gbias_ = sum_row( out_.mat() );
            // backprop
            if( is_firstlayer ){
                in_.mat() = dot( out_.mat(), wmat_.T() );
            }
        }
        virtual void GetUpdaters(std::vector<IUpdater*> &updaters){
            //TODO
        }
        virtual void SetParam(const char *name, const char* val){
            //TODO
        }
        virtual void InitModel(void){
            // TODO: how to add random here
            gwmat_ = 0.0f; gbias_ = 0.0f;
        }
        virtual void SaveModel(mshadow::utils::IStream &fo) const{
            wmat_.SaveBinary( fo );
            bias_.SaveBinary( fo );
            gwmat_.SaveBinary( fo );
            gbias_.SaveBinary( fo );
        }
        /*!
         * \brief Load model from binary file
         * \param fi input stream
         */
        virtual void LoadModel(mshadow::utils::IStream &fi){
            wmat_.LoadBinary( fi );
            bias_.LoadBinary( fi );
            gwmat_.LoadBinary( fi );
            gbias_.LoadBinary( fi );
        }
    private:        
        /*! \brief input node */
        Node<xpu> &in_; 
        /*! \brief output node */
        Node<xpu> &out_;
        /*! \brief weight matrix */
        mshadow::TensorContainer<xpu,2> wmat_;
        /*! \brief bias */
        mshadow::TensorContainer<xpu,1> bias_;
        /*! \brief weight matrix */
        mshadow::TensorContainer<xpu,2> gwmat_;
        /*! \brief bias */
        mshadow::TensorContainer<xpu,1> gbias_;
    };
}; // namespace cxxnet

#endif // CXXNET_LAYER_INL_HPP

