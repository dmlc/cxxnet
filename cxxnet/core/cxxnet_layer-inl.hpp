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

#if CXXNET_ADAPT_CAFFE
#include "cxxnet_caffe_adapter-inl.hpp"
#endif

namespace cxxnet{
    template<typename xpu>
    inline ILayer* CreateLayer_( int type, mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out );
};

namespace cxxnet{
    // expr is needed to use expression
    using namespace mshadow::expr;
    using namespace mshadow::utils;
    // Random init method
    using namespace cxxnet::rnd_type;
    /*! \brief potential parameters for each layer */
    struct LayerParam{
        /*! \brief number of hidden layers */
        int num_hidden;
        /*! \brief initialization sd for weight */
        float init_sigma;
        /*! \brief initialization random type */
        int random_type;
        /*! \brief number of output channel */
        int num_channel;
        /*! \brief number of parallel group */
        int num_group;
        /*! \brief kernel size */
        int kernel_size;
        /*! \brief stride prameter */
        int stride;
        /*! \brief padding */
        int pad;
        /*! \brief whether not include bias term */
        int no_bias;
        /*! \brief dropout threshold  */
        float dropout_threshold;
        LayerParam( void ){
            init_sigma = 0.01f;
            num_hidden = 0;
            random_type = 0;
            num_channel = 0;
            num_group = 1;
            kernel_size = 0;
            stride = 1;
            pad = 0;
            dropout_threshold = 0.0f;
            no_bias = 0;
        }
        /*!
         * \brief Set param for the layer from string
         * \param name parameter name
         * \param val string for configuration
         */
        inline void SetParam(const char *name, const char* val) {
            if( !strcmp( name, "init_sigma") )  init_sigma = (float)atof(val);
            if( !strcmp( name, "nhidden") )     num_hidden = atoi(val);
            if( !strcmp( name, "random_type"))  random_type = atoi(val);
            if( !strcmp( name, "nchannel") )    num_channel = atoi(val);
            if( !strcmp( name, "num_group") )   num_group = atoi(val);
            if( !strcmp( name, "kernel_size") ) kernel_size = atoi(val);
            if( !strcmp( name, "stride") )      stride      = atoi(val);
            if( !strcmp( name, "pad") )         pad      = atoi(val);
            if( !strcmp( name, "no_bias") )     no_bias = atoi(val);
            if( !strcmp( name, "threshold"))    dropout_threshold = (float)atof(val);
        }
    };
};

namespace cxxnet {
    /*! \brief some common edge operations, so far it is initlaization, maybe move as common functions */
    template<typename xpu>
    class EdgeLayer : public ILayer {
    public:
        /*!\brief every virtual class must have virtual destructor */
        virtual ~EdgeLayer( void ){}
    protected:
        /*! \brief constructor */
        EdgeLayer(mshadow::Random<xpu> &rnd ) : rnd_(rnd) {}
        /*! \brief Gaussian initialization */
        template<int dim>
        inline void InitGaussian(mshadow::TensorContainer<xpu, dim> &mat, real_t sigma) {
            rnd_.SampleGaussian(mat, 0.0f, sigma);
        }
        /*! \brief Uniform initialization */
        template<int dim>
        inline void InitUniform(mshadow::TensorContainer<xpu, dim> &mat, real_t a, real_t b) {
            rnd_.SampleUniform(mat, a, b);
        }
        /*! \brief Xavier initialization */
        template<int dim>
        inline void InitXavier(mshadow::TensorContainer<xpu,dim> &mat, index_t in_num, index_t out_num) {
            real_t a = sqrt(3.0f / (in_num + out_num));
            this->InitUniform(mat, -a, a);
        }
        /*! \brief random number generator */
        mshadow::Random<xpu> &rnd_;
    }; // class EdgeLayer
}; // namespace cxxnet

namespace cxxnet {
    // simple fully connected layer that connects two nodes
    template<typename xpu>
    class FullConnectLayer : public EdgeLayer<xpu> {
    public:
        FullConnectLayer( mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out )
            : EdgeLayer<xpu>(rnd), in_(in), out_(out) {}
        virtual ~FullConnectLayer( void ){}
        virtual void Forward(bool is_train) {
            this->Forward( wmat_ );
        }
        virtual void Backprop(bool prop_grad){
            this->Backprop( prop_grad, wmat_ );
        }
        virtual void AdjustNodeShape( void ) {
            Assert( in_.is_mat(), "input need to be a matrix" );
            Assert( param_.num_hidden > 0, "must set nhidden correctly" );
            out_.data.shape = mshadow::Shape4( 1, 1, in_.data.shape[1], param_.num_hidden );
        }
        virtual void GetUpdaters( const char *updater, std::vector<IUpdater*> &updaters ){
            updaters.push_back( CreateUpdater( updater, EdgeLayer<xpu>::rnd_, wmat_, gwmat_, "wmat" ) );
            if( param_.no_bias == 0 ){
                updaters.push_back( CreateUpdater( updater, EdgeLayer<xpu>::rnd_, bias_, gbias_, "bias" ) );
            }
        }
        virtual void SetParam(const char *name, const char* val){
            param_.SetParam( name, val );
        }
        virtual void InitModel(void){
            // resize to correct shape
            wmat_.Resize( mshadow::Shape2( out_.data.shape[0], in_.data.shape[0] ) );
            gwmat_.Resize( wmat_.shape );
            bias_.Resize( mshadow::Shape1( out_.data.shape[0] ) );
            gbias_.Resize( bias_.shape );
            // random initalize
            if (param_.random_type == kGaussian) {
                EdgeLayer<xpu>::template InitGaussian<2> (wmat_, param_.init_sigma);
            } else {
                EdgeLayer<xpu>::template InitXavier<2> (wmat_, wmat_.shape[0], wmat_.shape[1]);
            }
            bias_ = 0.0f; gwmat_ = 0.0f; gbias_ = 0.0f;
        }
        virtual void SaveModel(mshadow::utils::IStream &fo) const{
            fo.Write( &param_, sizeof(LayerParam) );
            wmat_.SaveBinary( fo );
            bias_.SaveBinary( fo );
        }
        virtual void LoadModel(mshadow::utils::IStream &fi){
            Assert( fi.Read( &param_, sizeof(LayerParam) ) != 0, "load model");
            wmat_.LoadBinary( fi );
            bias_.LoadBinary( fi );
        }
    protected:
        inline void Forward(mshadow::Tensor<xpu,2> wmat) {
            index_t nbatch = in_.data.shape[1];
            out_.mat()  = dot( in_.mat(), wmat.T() );
            if( param_.no_bias == 0 ){
                out_.mat() += repmat( bias_, nbatch );
            }
        }
        inline void Backprop(bool prop_grad, mshadow::Tensor<xpu,2> wmat){
            gwmat_ = dot( out_.mat().T(), in_.mat() );
            if( param_.no_bias == 0 ){
                gbias_ = sum_rows( out_.mat() );
            }
            // backprop
            if( prop_grad ){
                in_.mat() = dot( out_.mat(), wmat );
            }
        }
    protected:
        /*! \brief parameters that potentially be useful */
        LayerParam param_;
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

    /*! \brief softmax layer, do softmax transformation during forward */
    template<typename xpu>
    class SoftmaxLayer: public ILayer{
    public:
        SoftmaxLayer( mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out )
            :out_(out){
            Assert( &in == &out, "softmax layer must self loop e.g layer[1->1] = softmax" );
        }
        virtual ~SoftmaxLayer( void ){}
        virtual void Forward(bool is_train){
            mshadow::Softmax( out_.mat(), out_.mat() );
        }
        virtual void Backprop(bool prop_grad){
            index_t nbatch =  out_.mat().shape[1];
            out_.mat() *= 1.0f / nbatch;
        }
    private:
        /*! \brief only transform on out */
        Node<xpu> &out_;
    };

    template<typename xpu>
    class ConvolutionLayer : public EdgeLayer<xpu> {
    public:
        ConvolutionLayer( mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out )
            : EdgeLayer<xpu>(rnd), in_(in), out_(out) {
        }
        virtual ~ConvolutionLayer( void ){}
        virtual void Forward(bool is_train) {
            const index_t nbatch = in_.data.shape[3];

            for( index_t i = 0; i < nbatch; ++ i ){
                if( param_.pad == 0 ){
                    temp_col_ = unpack_patch2col( in_.data[i], param_.kernel_size, param_.stride );
                }else{
                    temp_col_ = unpack_patch2col( pad(in_.data[i],param_.pad), param_.kernel_size, param_.stride );
                }

                const index_t gstride = temp_col_.shape[1] / param_.num_group;
                for( int gid = 0; gid < param_.num_group; ++ gid ){
                    mshadow::Tensor<xpu,2> tmpc = temp_col_.Slice( gstride*gid, gstride*(gid+1) );
                    temp_dst_[ gid ] = dot( wmat_[gid], tmpc );
                }
                out_.data[i] = reshape( temp_dst_, out_.data[i].shape );
            }
            if( param_.no_bias == 0 ){
                // add bias, broadcast bias to dim 2: channel
                out_.data += broadcast<2>( bias_, out_.data.shape );
            }
        }
        virtual void Backprop(bool prop_grad){
            const index_t nbatch = in_.data.shape[3];

            if( param_.no_bias == 0 ){
                gbias_ = sumall_except_dim<2>( out_.data );
            }
            gwmat_ = 0.0f;
            for( index_t i = 0; i < nbatch; ++ i ){
                temp_dst_ = reshape( out_.data[i], temp_dst_.shape );
                if( param_.pad == 0 ){
                    temp_col_ = unpack_patch2col( in_.data[i], param_.kernel_size, param_.stride );
                }else{
                    temp_col_ = unpack_patch2col( pad(in_.data[i],param_.pad), param_.kernel_size, param_.stride );
                }

                const index_t gstride = temp_col_.shape[1] / param_.num_group;
                for( int gid = 0; gid < param_.num_group; ++ gid ){
                    mshadow::Tensor<xpu,2> tmpc = temp_col_.Slice( gstride*gid, gstride*(gid+1) );
                    gwmat_[gid] += dot( temp_dst_[gid], tmpc.T() );
                }

                if( prop_grad ){
                    for( int gid = 0; gid < param_.num_group; ++ gid ){
                        mshadow::Tensor<xpu,2> tmpc = temp_col_.Slice( gstride*gid, gstride*(gid+1) );
                        tmpc = dot( wmat_[gid].T(), temp_dst_[gid] );
                    }
                    if( param_.pad == 0 ){
                        in_.data[i] = pack_col2patch( temp_col_, in_.data[i].shape, param_.kernel_size, param_.stride );
                    }else{
                        mshadow::Shape<3> pshape = in_.data[i].shape; pshape[0] += 2*param_.pad; pshape[1] += 2*param_.pad;
                        in_.data[i] = unpad( pack_col2patch( temp_col_, pshape, param_.kernel_size, param_.stride ), param_.pad );
                    }
                }
            }
        }
        virtual void AdjustNodeShape( void ) {
            const index_t ksize   = static_cast<index_t>( param_.kernel_size );
            const index_t kstride = static_cast<index_t>( param_.stride );
            Assert( in_.data.shape[2] % param_.num_group == 0,  "input channels must divide group size" );
            Assert( param_.num_channel % param_.num_group == 0, "output channels must divide group size" );
            Assert( param_.num_channel > 0, "must set nchannel correctly" );
            Assert( param_.kernel_size > 0, "must set kernel_size correctly" );
            Assert( ksize <= in_.data.shape[0] && ksize <= in_.data.shape[1], "kernel size exceed input" );

            mshadow::Shape<4> oshape = mshadow::
                Shape4( in_.data.shape[3], param_.num_channel,
                        (in_.data.shape[1] + 2 * param_.pad - ksize)/kstride + 1,
                        (in_.data.shape[0] + 2 * param_.pad - ksize)/kstride + 1 );
            out_.data.shape = oshape;

            // helper structure
            temp_col_.Resize( mshadow::Shape2( in_.data.shape[2]*ksize*ksize, oshape[1]*oshape[0] ) );
            temp_dst_.Resize( mshadow::Shape3( param_.num_group, param_.num_channel/param_.num_group, oshape[1]*oshape[0] ) );
        }
        virtual void GetUpdaters( const char *updater, std::vector<IUpdater*> &updaters ){
            updaters.push_back( CreateUpdater( updater, EdgeLayer<xpu>::rnd_, wmat_, gwmat_, "wmat" ) );
            if( param_.no_bias == 0 ){
                updaters.push_back( CreateUpdater( updater, EdgeLayer<xpu>::rnd_, bias_, gbias_, "bias" ) );
            }
        }
        virtual void SetParam(const char *name, const char* val){
            param_.SetParam( name, val );
        }
        virtual void InitModel(void){
            // resize to correct shape, use 2d to store the weight, since we use dot
            wmat_.Resize( mshadow::Shape3( param_.num_group, param_.num_channel / param_.num_group,
                                           in_.data.shape[2] / param_.num_group * param_.kernel_size * param_.kernel_size ) );
            gwmat_.Resize( wmat_.shape );
            bias_.Resize( mshadow::Shape1( param_.num_channel ) );
            gbias_.Resize( bias_.shape );

            if (param_.random_type == kGaussian) {
                EdgeLayer<xpu>::template InitGaussian<3> (wmat_, param_.init_sigma);
            } else {
                EdgeLayer<xpu>::template InitXavier<3> (wmat_, wmat_.shape[1], wmat_.shape[0]);
            }

            bias_ = 0.0f; gwmat_ = 0.0f; gbias_ = 0.0f;
        }
        virtual void SaveModel(mshadow::utils::IStream &fo) const{
            fo.Write( &param_, sizeof(LayerParam) );
            wmat_.SaveBinary( fo );
            bias_.SaveBinary( fo );
        }
        virtual void LoadModel(mshadow::utils::IStream &fi){
            Assert( fi.Read( &param_, sizeof(LayerParam) ) != 0, "load model");
            wmat_.LoadBinary( fi );
            bias_.LoadBinary( fi );
        }
    private:
        /*! \brief parameters that potentially be useful */
        LayerParam param_;
        /*! \brief input node */
        Node<xpu> &in_;
        /*! \brief output node */
        Node<xpu> &out_;
        /*! \brief weight matrix */
        mshadow::TensorContainer<xpu,3> wmat_;
        /*! \brief bias */
        mshadow::TensorContainer<xpu,1> bias_;
        /*! \brief accumulates the gradient of weight matrix */
        mshadow::TensorContainer<xpu,3> gwmat_;
        /*! \brief accumulates the gradient of bias */
        mshadow::TensorContainer<xpu,1> gbias_;
        /*! \brief temporary data structure to store patches */
        mshadow::TensorContainer<xpu,2> temp_col_;
        /*! \brief temporary data structure to store results */
        mshadow::TensorContainer<xpu,3> temp_dst_;
    };

    template<typename Reducer, bool scalebysize, typename xpu>
    class PoolingLayer : public ILayer {
    public:
        PoolingLayer(Node<xpu> &in, Node<xpu> &out)
            : in_(in), out_(out) {
        }
        virtual ~PoolingLayer() {}
        virtual void Forward(bool is_train) {
            const int ksize = param_.kernel_size;
            if( !scalebysize ){
                tmp_ = pool<Reducer>(in_.data, ksize, param_.stride);
            }else{                
                tmp_ = pool<Reducer>(in_.data, ksize, param_.stride) * (1.0f/(ksize*ksize) );
            }
            mshadow::Copy( out_.data, tmp_ );
        }
        virtual void Backprop(bool prop_grad) {
            if (prop_grad) {
                const int ksize = param_.kernel_size;
                if( !scalebysize ){
                    in_.data = unpool<Reducer>(in_.data, tmp_, out_.data, param_.kernel_size, param_.stride);
                }else{
                    in_.data = unpool<Reducer>(in_.data, tmp_, out_.data, param_.kernel_size, param_.stride) * (1.0f/(ksize*ksize) );
                }
            }
        }
        virtual void SetParam(const char *name, const char* val) {
            param_.SetParam( name, val );
        }
        virtual void AdjustNodeShape() {
            const index_t ksize   = static_cast<index_t>( param_.kernel_size );
            const index_t kstride = static_cast<index_t>( param_.stride );
            Assert( param_.kernel_size > 0, "must set kernel_size correctly" );
            Assert( ksize <= in_.data.shape[0] && ksize <= in_.data.shape[1], "kernel size exceed input" );
            mshadow::Shape<4> oshape = mshadow::
                Shape4( in_.data.shape[3], in_.data.shape[2],
                        (in_.data.shape[1] - ksize)/kstride + 1,
                        (in_.data.shape[0] - ksize)/kstride + 1 );
            tmp_.Resize( oshape ); out_.data.shape = oshape;
        }
    private:
        /*! \brief parameters that potentially be useful */
        LayerParam param_;
        /*! \brief input node */
        Node<xpu> &in_;
        /*! \brief output node */
        Node<xpu> &out_;
        /*! \brief pooled result */
        mshadow::TensorContainer<xpu, 4> tmp_;
    }; // class PoolingLayer


    template<typename xpu,typename ForwardOp, typename BackOp >
    class ActivationLayer : public ILayer{
    public:
        ActivationLayer( Node<xpu> &in, Node<xpu> &out )
            :in_(in), out_(out) {
        }
        virtual ~ActivationLayer( void ){}
        virtual void Forward( bool is_train ) {
            in_.data = F<ForwardOp>( in_.data );
            mshadow::Copy( out_.data, in_.data );
        }
        virtual void Backprop( bool prop_grad ){
            in_.data = F<BackOp>( in_.data ) * out_.data;
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
                in_.data = unpad(out_.data, pad_);
            }
        }
        virtual void AdjustNodeShape() {
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
    // out[i] = in[i] * (sum_{j\in N(i)} in[j]^2 + alpha)^{-beta}
    // gradient first part : gradout[i] * (sum_{j\in N(i)} in[j]^2 + alpha)^{-beta}
    // gradient second part: 2 * in[i] * ( (sum_{j\in N(i)} gradout[j] / (sum_{jj\in N(j)} in[jj]^2 + alpha)^{-beta-1}) *(-beta) )
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
        virtual void AdjustNodeShape( void ) {            
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
    template<typename xpu>
    class FlattenLayer : public ILayer{
    public:
        FlattenLayer( Node<xpu> &in, Node<xpu> &out )
            :in_(in), out_(out) {
        }
        virtual ~FlattenLayer( void ){}
        virtual void Forward( bool is_train ) {
            out_.data = reshape( in_.data, out_.data.shape );
        }
        virtual void Backprop( bool prop_grad ){
            if( prop_grad ){
                in_.data = reshape( out_.data, in_.data.shape );
            }
        }
        virtual void AdjustNodeShape( void ) {
            mshadow::Shape<4> ishape = in_.data.shape;
            out_.data.shape = mshadow::Shape4( 1, 1, ishape[3], ishape[2]*ishape[1]*ishape[0] );
        }
    private:
        /*! \brief input node */
        Node<xpu> &in_;
        /*! \brief output node */
        Node<xpu> &out_;
    };
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
                Parent::rnd_.SampleUniform(mask_, 0.0f, 1.0f);
                mask_ = F<op::threshold>(mask_, pkeep);
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
        virtual void AdjustNodeShape( void ){
            this->mask_.Resize( mshadow::Shape2( this->out_.data.shape[0], this->in_.data.shape[0] ) );
        }
    private:
        typedef FullConnectLayer<xpu> Parent;
    private:
        mshadow::TensorContainer<xpu, 2> mask_, tmpw_;
    }; // class DropconnLayer

    template<typename xpu>
    class DropoutLayer : public ILayer {
    public:
        DropoutLayer(mshadow::Random<xpu> &rnd, Node<xpu> &in, Node<xpu> &out)
            :rnd_(rnd), out_(out) {
            Assert( &in == &out, "dropout layer must self loop e.g layer[1->1] = dropout" );
        }
        virtual void SetParam(const char *name, const char* val){
            param_.SetParam( name, val );
        }
        virtual void Forward( bool is_train ) {
            if (is_train) {
                const real_t pkeep = 1.0f - param_.dropout_threshold;
                // mask_ = F<op::threshold>(rnd_.uniform(), pkeep ) * (1.0f/pkeep);
                rnd_.SampleUniform(mask_, 0.0f, 1.0f);
                mask_ = F<op::threshold>(mask_, pkeep);
                out_.data = out_.data * mask_;
            }
        }
        virtual void Backprop( bool prop_grad ) {
            if (prop_grad) {
                out_.data *= mask_;
            }
        }
        virtual void AdjustNodeShape( void ) {
            utils::Assert(param_.dropout_threshold >= 0.0f && param_.dropout_threshold < 1.0f, "DropoutLayer: invalid dropout threshold\n");
            mask_.Resize( out_.data.shape );
        }
    private:
        /*! \brief output node */
        Node<xpu> &out_;
        /*! \brief dropout mask */
        mshadow::TensorContainer<xpu, 4> mask_;
        /*! \brief random number generator */
        mshadow::Random<xpu> &rnd_;
        /*! \brief parameters that potentially be useful */
        LayerParam param_;
    }; // class DropoutLayer
}; // namespace cxxnet

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
            virtual void Update( void ){
                umaster_->Update();  uslave_->Update();
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
        virtual void AdjustNodeShape( void ){
            slv_in_.data.shape = in_.data.shape;
            master_->AdjustNodeShape();
            slave_->AdjustNodeShape();
            utils::Assert( slv_out_.data.shape == out_.data.shape, "PairTestLayer:AdjustNodeShape mismatch" );
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
        if( !strncmp( type, "pairtest-", 9 ) ){
            char tmaster[256], tslave[256];
            sscanf( type + 9, "%[^-]-%[^:]", tmaster, tslave );
            return GetLayerType(tmaster) * 1000  + GetLayerType(tslave);
        }
        using namespace layer_type;
        if( !strcmp( type, "fullc") )   return kFullConnect;
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
