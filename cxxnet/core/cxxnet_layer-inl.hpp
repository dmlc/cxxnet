#ifndef CXXNET_LAYER_INL_HPP
#define CXXNET_LAYER_INL_HPP
#pragma once
/*!
 * \file cxxnet_layer-inl.hpp
 * \brief implementation of different layers
 * \author Bing Xu, Tianqi Chen
 */

#include "cxxnet_core.h"
#include "cxxnet_op.h"
#include "mshadow/tensor_container.h"

namespace cxxnet{
    // expr is needed to use expression
    using namespace mshadow::expr;
    using namespace mshadow::utils;

    /*! \brief potential parameters for each layer */
    struct LayerParam{
        /*! \brief number of hidden layers */
        int num_hidden;
        /*! \brief initialization sd for weight */
        float init_sigma;
        LayerParam( void ){
            init_sigma = 0.01f;
            num_hidden = 0;
        }
        /*!
         * \brief Set param for the layer from string
         * \param name parameter name
         * \param val string for configuration
         */
        inline void SetParam(const char *name, const char* val) {
            if( !strcmp( name, "init_sigma") ) init_sigma = (float)atof(val);
            if( !strcmp( name, "nhidden") ) num_hidden = atoi(val);
        }
    };
};

namespace cxxnet {
    // simple fully connected layer that connects two nodes
    template<typename xpu>
    class FullConnectLayer : public ILayer{
    public:
        FullConnectLayer( mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out )
            :rnd_(rnd), in_(in), out_(out){

        }
        virtual ~FullConnectLayer( void ){
        }
        virtual void Forward(bool is_train) {
            index_t nbatch = in_.data.shape[1];
            out_.mat()  = dot( in_.mat(), wmat_ );
            out_.mat() += repmat( bias_, nbatch );
        }
        virtual void Backprop(bool is_firstlayer){
            index_t nbatch = in_.data.shape[1];
            real_t scale = 1.0f / nbatch;
            // accumulates gradient, instead of set gradient
            gwmat_ += scale * dot( in_.mat().T(), out_.mat() );
            gbias_ += scale * sum_rows( out_.mat() );
            // backprop
            if( is_firstlayer ){
                in_.mat() = dot( out_.mat(), wmat_.T() );
            }
        }
        virtual void AdjustNodeShape( void ) {
            Assert( in_.is_mat(), "input need to be a matrix" );
            Assert( param_.num_hidden > 0, "must set nhidden correctly" );
            out_.data.shape = mshadow::Shape4( 1, 1, in_.data.shape[1], param_.num_hidden );
        }
        virtual void GetUpdaters( const char *updater, std::vector<IUpdater*> &updaters ){
            updaters.push_back( CreateUpdater( updater, rnd_, wmat_, gwmat_, "wmat" ) );
            updaters.push_back( CreateUpdater( updater, rnd_, bias_, gbias_, "bias" ) );
        }
        virtual void SetParam(const char *name, const char* val){
            param_.SetParam( name, val );
        }
        virtual void InitModel(void){
            // resize to correct shape
            wmat_.Resize( mshadow::Shape2( in_.data.shape[0], out_.data.shape[0] ) );
            gwmat_.Resize( wmat_.shape );
            bias_.Resize( mshadow::Shape1( out_.data.shape[0] ) );
            gbias_.Resize( bias_.shape );
            // random initalize
            rnd_.SampleGaussian( wmat_, 0.0f, param_.init_sigma );
            bias_ = 0.0f; gwmat_ = 0.0f; gbias_ = 0.0f;
        }
        virtual void SaveModel(mshadow::utils::IStream &fo) const{
            fo.Write( &param_, sizeof(LayerParam) );
            wmat_.SaveBinary( fo );
            bias_.SaveBinary( fo );
            gwmat_.SaveBinary( fo );
            gbias_.SaveBinary( fo );
        }
        virtual void LoadModel(mshadow::utils::IStream &fi){
            Assert( fi.Read( &param_, sizeof(LayerParam) ) != 0, "load model");
            wmat_.LoadBinary( fi );
            bias_.LoadBinary( fi );
            gwmat_.LoadBinary( fi );
            gbias_.LoadBinary( fi );
        }
    private:
        /*! \brief parameters that potentially be useful */
        LayerParam param_;
        /*! \brief random number generator */
        mshadow::Random<xpu> &rnd_;
        /*! \brief input node */
        Node<xpu> &in_;
        /*! \brief output node */
        Node<xpu> &out_;
        /*! \brief weight matrix */
        mshadow::TensorContainer<xpu,2> wmat_;
        /*! \brief bias */
        mshadow::TensorContainer<xpu,1> bias_;
        /*! \brief accumulates the gradient of weight matrix */
        mshadow::TensorContainer<xpu,2> gwmat_;
        /*! \brief accumulates the gradient of bias */
        mshadow::TensorContainer<xpu,1> gbias_;
    };

    // For softmax, we do not need to store weight/bias, only use softmax as a kinda of transformation
    // we can use full layer -> softmax layer
    template<typename xpu>
    class SoftmaxLayer: public ILayer{
    public:
        SoftmaxLayer( mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out )
            :out_(out){
            Assert( &in == &out, "softmax layer must self loop e.g layer[1->1] = softmax" );
        }
        virtual void Forward(bool is_train){
            mshadow::Softmax( out_.mat(), out_.mat() );
        }
        virtual void Backprop(bool is_firstlayer){
            // do nothing
        }
    private:
        /*! \brief only transform on out */
        Node<xpu> &out_;
    };
}; // namespace cxxnet

namespace cxxnet {

    template<typename xpu,typename ForwardOp, typename BackOp >
    class ActivationLayer : public ILayer{
    public:
        ActivationLayer( Node<xpu> &in, Node<xpu> &out )
            :in_(in), out_(out) {
        }
        virtual ~ActivationLayer( void ){}
        virtual void Forward( bool is_train ) {
            in_.mat() = F<ForwardOp>( in_.mat() );
            mshadow::Copy( out_.mat(), in_.mat() );
        }
        virtual void Backprop( bool is_firstlayer ){
            in_.mat() = F<BackOp>( in_.mat() ) * out_.mat();
        }
        virtual void AdjustNodeShape( void ) {
            out_.data.shape = in_.data.shape;
        }
    private:
        /*! \brief input node */
        Node<xpu> &in_;
        /*! \brief output node */
        Node<xpu> &out_;
    };
};

namespace cxxnet{
    /* layer patch that handles memory issues */
    template<typename xpu>
    struct LayerPatch: public ILayer{
    public:
        LayerPatch( ILayer *base, Node<xpu>& in, Node<xpu>& out )
            :base_(base), in_(in), out_(out){}
        virtual ~LayerPatch( void ){ delete base_; }
        virtual void Forward( bool is_train ){
            in_.Pin(); out_.Pin();
            base_->Forward( is_train );
            in_.Unpin(); out_.Unpin();
        }
        virtual void Backprop( bool is_firstlayer ){
            in_.Pin(); out_.Pin();
            base_->Backprop( is_firstlayer );
            in_.Unpin(); out_.Unpin();
        }
    public:
        virtual void AdjustNodeShape( void ){
            base_->AdjustNodeShape();
        }
        virtual void GetUpdaters( const char *updater, std::vector<IUpdater*> &updaters ) {
            base_->GetUpdaters( updater, updaters );
        }
        virtual void SetParam( const char *name, const char* val ) {
            base_->SetParam( name, val );
        }
        virtual void InitModel(void) {
            base_->InitModel();
        }
        virtual void SaveModel(mshadow::utils::IStream &fo) const {
            base_->SaveModel( fo );
        }
        virtual void LoadModel(mshadow::utils::IStream &fi) {
            base_->LoadModel( fi );
        }
    private:
        ILayer *base_;
        Node<xpu> &in_, &out_;
    };
};

namespace cxxnet{
    inline int GetLayerType( const char *type ){
        using namespace layer_type;
        if( !strcmp( type, "fullc") )   return kFullConnect;
        if( !strcmp( type, "softmax") ) return kSoftmax;
        if( !strcmp( type, "relu") ) return kRectifiedLinear;
        if( !strcmp( type, "sigmoid") ) return kSigmoid;
        if( !strcmp( type, "tanh") ) return kTanh;
        return 0;
    }

    template<typename xpu>
    inline ILayer* CreateLayer_( int type, mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out ){
        using namespace layer_type;
        switch( type ){
        case kFullConnect: return new FullConnectLayer<xpu>( rnd, in, out );
        case kSoftmax    : return new SoftmaxLayer<xpu>( rnd, in, out );
        case kSigmoid : return new ActivationLayer<xpu,op::sigmoid,op::sigmoid_grad>(in, out);
        case kTanh    : return new ActivationLayer<xpu,op::tanh,op::tanh_grad>(in, out);
        case kRectifiedLinear: return new ActivationLayer<xpu,op::relu,op::relu_grad>(in, out);
        default: Error("unknown layer type");
        }
        return NULL;
    }
    template<typename xpu>
    inline ILayer* CreateLayer( int type, mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out ){
        return new LayerPatch<xpu>( CreateLayer_<xpu>(type,rnd,in,out), in, out );
    }
    template<typename xpu>
    inline ILayer* CreateLayer( const char *type, mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out ){
        return CreateLayer( GetLayerType(type), rnd, in, out );
    }
};


#endif // CXXNET_LAYER_INL_HPP
