#ifndef CXXNET_LAYER_INL_HPP_
#define CXXNET_LAYER_INL_HPP_
/*!
 * \file layer-inl.hpp
 * \brief implementation of different layers
 * \author Bing Xu, Tianqi Chen
 */
#include "cxxnet_core.h"
#include "cxxnet_op.h"




namespace cxxnet {
    // simple fully connected layer that connects two nodes





};

namespace cxxnet {
    template<typename xpu>
    class PaddingLayer : public ILayer {
    public:
        PaddingLayer(Node<xpu> &in, Node<xpu> &out)
            : in_(in), out_(out) {
        }
        virtual ~PaddingLayer(){}
        virtual void SetParam(const char *name, const char *val) {
            if (!strcmp(name, "pad")) pad_ = static_cast<index_t>(atoi(val));
        }
        virtual void Forward(bool is_train) {
            out_.data = pad(in_.data, pad_);
        }
        virtual void Backprop(bool prop_grad) {
            if (prop_grad) {
                in_.data = crop(out_.data, in_.data[0][0].shape);
            }
        }
        virtual void InitLayer() {
            mshadow::Shape<4> oshape = mshadow::Shape4(in_.data.shape[3], in_.data.shape[2],
                                              in_.data.shape[1] + 2 * pad_,
                                              in_.data.shape[0] + 2 * pad_);
            out_.data.shape = oshape;
        }
    private:
        /*! \brief input node */
        Node<xpu> &in_;
        /*! \brief output node */
        Node<xpu> &out_;
        /*! \brief padding size */
        index_t pad_;
    }; // class padding layer
}; // namespace cxxnet

namespace cxxnet{
    template<typename xpu>
    class LRNLayer : public ILayer {
    public:
        LRNLayer(Node<xpu> &in, Node<xpu> &out)
            : in_(in), out_(out) {
            // default values
            this->knorm_ = 1.0f;
            this->nsize_ = 3;
        }
        virtual ~LRNLayer( void ){}
        virtual void SetParam(const char *name, const char *val) {
            if (!strcmp(name, "local_size")) nsize_ = static_cast<index_t>( atoi(val) );
            if (!strcmp(name, "alpha"))      alpha_ = static_cast<real_t>( atof(val) );
            if (!strcmp(name, "beta"))       beta_  = static_cast<real_t>( atof(val) );
            if (!strcmp(name, "knorm"))      knorm_ = static_cast<real_t>( atof(val) );
        }
        virtual void Forward(bool is_train) {
            using namespace mshadow;
            const real_t salpha = alpha_ / nsize_;
            // stores normalizer without power
            tmp_norm = chpool<red::sum>( F<op::square>( in_.data ) , nsize_ ) * salpha + knorm_;
            out_.data = in_.data * F<op::power>( tmp_norm, -beta_ );
        }
        virtual void Backprop(bool prop_grad) {
            using namespace mshadow;
            const real_t salpha = alpha_ / nsize_;
            if( prop_grad ) {
                // backup input data
                mshadow::Copy( tmp_in, in_.data );
                // first gradient to a[i], will be 1 / normalizer
                in_.data = out_.data * F<op::power>( tmp_norm, -beta_ );
                // gradient to normalizer
                in_.data += ( - 2.0f * beta_ * salpha ) * chpool<red::sum>( out_.data * tmp_in * F<op::power>( tmp_norm, -beta_-1.0f ), nsize_ )  * tmp_in;
            }
        }
        virtual void InitLayer( void ) {
            out_.data.shape = in_.data.shape;
            tmp_in.Resize( in_.data.shape );
            tmp_norm.Resize( in_.data.shape );
        }
    private:
        /*! \brief input node */
        Node<xpu> &in_;
        /*! \brief output node */
        Node<xpu> &out_;
        /*! \brief input temp data */
        mshadow::TensorContainer<xpu,4> tmp_in;
        /*! \brief temp normalizer */
        mshadow::TensorContainer<xpu,4> tmp_norm;
        /*! \brief alpha */
        real_t alpha_;
        /*! \brief beta */
        real_t beta_;
        /*! \brief knorm */
        real_t knorm_;
        /*! \brief neighbor size*/
        index_t nsize_;
    }; // class lrn layer
}; // namespace cxxnet

namespace cxxnet{

};

namespace cxxnet {
    template<typename xpu>
    class DropConnLayer : public FullConnectLayer<xpu> {
    public:
        DropConnLayer(mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out)
            : Parent(rnd, in, out) {}
        virtual void Forward(bool is_train) {
            if( is_train ){
                const real_t pkeep = 1.0f - Parent::param_.dropout_threshold;
                mask_ = F<op::threshold>( Parent::rnd_.uniform( mask_.shape ), pkeep ) * (1.0f/pkeep);
                tmpw_ = this->wmat_ * mask_;
            }else{
                mshadow::Copy( tmpw_, this->wmat_ );
            }
            Parent::Forward( tmpw_ );
        }
        virtual void Backprop(bool prop_grad) {
            Parent::Backprop( prop_grad, tmpw_ );
            Parent::gwmat_ *= mask_;
        }
        virtual void InitLayer( void ){
            this->mask_.Resize( mshadow::Shape2( this->out_.data.shape[0], this->in_.data.shape[0] ) );
        }
    private:
        typedef FullConnectLayer<xpu> Parent;
    private:
        mshadow::TensorContainer<xpu, 2> mask_, tmpw_;
    }; // class DropconnLayer


}; // namespace cxxnet

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
        virtual void Backprop( bool prop_grad ){
            in_.Pin(); out_.Pin();
            base_->Backprop( prop_grad );
            in_.Unpin(); out_.Unpin();
        }
    public:
        virtual void InitLayer( void ){
            base_->InitLayer();
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
        if( !strncmp( type, "pairtest-", 9 ) ){
            char tmaster[256], tslave[256];
            sscanf( type + 9, "%[^-]-%[^:]", tmaster, tslave );
            return GetLayerType(tmaster) * 1000  + GetLayerType(tslave);
        }
        using namespace layer_type;
        if( !strcmp( type, "fullc") )   return kFullConnect;
        if( !strcmp( type, "bias") )   return kBias;
        if( !strcmp( type, "softmax") ) return kSoftmax;
        if( !strcmp( type, "relu") ) return kRectifiedLinear;
        if( !strcmp( type, "sigmoid") ) return kSigmoid;
        if( !strcmp( type, "tanh") ) return kTanh;
        if( !strcmp( type, "softplus") ) return kSoftplus;
        if( !strcmp( type, "flatten") )  return kFlatten;
        if( !strcmp( type, "dropout") ) return kDropout;
        if( !strcmp( type, "dropconn") ) return kDropConn;
        if( !strcmp( type, "conv") )     return kConv;
        if( !strcmp( type, "max_pooling")) return kMaxPooling;
        if( !strcmp( type, "sum_pooling")) return kSumPooling;
        if( !strcmp( type, "avg_pooling")) return kAvgPooling;
        if( !strcmp( type, "padding"))   return kPadding;
        if( !strcmp( type, "lrn"))       return kLRN;
        if( !strcmp( type, "caffe") )    return kCaffe;
        fprintf(stderr, "unknown layer type: %s\n", type);
        Error("unknown layer type" );
        return 0;
    }

    template<typename xpu>
    inline ILayer* CreateLayer_( int type, mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out ){
        using namespace layer_type;
        if( type >= 1000 ){
            return new PairTestLayer<xpu>( rnd, in, out, type/1000, type % 1000);
        }
        switch( type ){
        case kFullConnect: return new FullConnectLayer<xpu>( rnd, in, out );
        case kBias: return new BiasLayer<xpu>( rnd, in, out );
        case kSoftmax    : return new SoftmaxLayer<xpu>( rnd, in, out );
        case kSigmoid : return new ActivationLayer<xpu,op::sigmoid,op::sigmoid_grad>(in, out);
        case kTanh    : return new ActivationLayer<xpu,op::tanh,op::tanh_grad>(in, out);
        case kRectifiedLinear: return new ActivationLayer<xpu,op::relu,op::relu_grad>(in, out);
        case kSoftplus: return new ActivationLayer<xpu,op::softplus,op::softplus_grad>(in, out);
        case kFlatten:  return new FlattenLayer<xpu>( in, out );
        case kDropConn: return new DropConnLayer<xpu>(rnd, in, out);
        case kDropout: return new DropoutLayer<xpu>(rnd, in, out);
        case kConv:    return new ConvolutionLayer<xpu>( rnd, in, out );
        case kMaxPooling: return new PoolingLayer<mshadow::red::maximum, false, xpu>(in, out);
        case kSumPooling: return new PoolingLayer<mshadow::red::sum, false, xpu>(in, out);
        case kAvgPooling: return new PoolingLayer<mshadow::red::sum, true, xpu>(in, out);
        case kPadding: return new PaddingLayer<xpu>(in, out);
        case kLRN:     return new LRNLayer<xpu>(in, out);
#if CXXNET_ADAPT_CAFFE
        case kCaffe: return new CaffeLayer<xpu>(rnd,in,out);
#endif
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
