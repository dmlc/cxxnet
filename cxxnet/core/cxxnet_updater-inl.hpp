#ifndef CXXNET_UPDATER_INL_HPP
#define CXXNET_UPDATER_INL_HPP
#pragma once
/*!
 * \file cxxnet_updater-inl.hpp
 * \brief implementation of updaters
 * \author Tianqi Chen
 */
#include <climits>
#include <string>
#include "cxxnet_core.h"
#include "mshadow/tensor_container.h"
#include "../utils/cxxnet_global_random.h"

namespace cxxnet{
    // expr is needed to use expression
    using namespace mshadow::expr;
    using namespace mshadow::utils;

    /*! \brief potential parameters for each layer */
    struct UpdaterParam{
        /*! \brief tag of current parameter group */
        std::string tag;
        /*! \brief current round */
        int round;
        /*! \brief whether can print messages */
        int silent;
        /*! \brief learning rate */
        float learning_rate;
        /*! \brief weight decay */
        float wd;
        /*! \brief momentum  */
        float momentum;
        // scheduling parameters
        /*! \brief type of learning rate schedule */
        int lr_schedule;
        /*! \brief base learning rate */
        float base_lr_;
        /*! \brief period of lr decay */
        long lr_step;
        /*! \brief decay parameter gamma */
        float lr_gamma;
        /*! \brief decay parameter gamma */
        float lr_alpha;
        /*! \brief decay parameter factor */
        float lr_factor;
        /*! \brief minimum learning rate */
        float lr_minimum;
        /*! \brief start scheduling epoch */
        long start_epoch;
        /*! \brief constructor that sets default parameters */
        UpdaterParam( void ){
            base_lr_ = 0.01f;
            lr_schedule = 0;
            lr_step = 1;
            lr_alpha = 0.5f;
            lr_gamma = 0.5f;
            lr_factor = 0.1f;
            lr_minimum = 0.00001;
            start_epoch = 0;
            wd = 0.0f;
            momentum = 0.9f;
            silent = 0;
        }
        /*! \brief do learning rate or other parameter schedule at round epoch */
        inline void ScheduleEpoch( long epoch ){
            if (epoch < start_epoch) {
                learning_rate = base_lr_;
                return;
            }
            switch( lr_schedule ){
            case 0: learning_rate = base_lr_; break;
            case 1: learning_rate = base_lr_ * powf( lr_gamma, epoch / lr_step ); break;
            case 2: learning_rate = base_lr_ * powf( 1.0f + (epoch/lr_step) * lr_gamma, -lr_alpha ); break;
            case 3: learning_rate = base_lr_ * powf(lr_factor, epoch / lr_step); break;
            default: utils::Error("unknown schedule type");
            }
            learning_rate = learning_rate < lr_minimum ? lr_minimum : learning_rate;
        }
        /*!
         * \brief Set param for the layer from string
         * \param name parameter name
         * \param val string for configuration
         */
        inline void SetParam( const char *name, const char* val ) {
            // if we set "bias:wd = 0.0", and tag == "bias", the it will set wd in current updater param
            // but will not affect settings with other tags
            if( !strncmp( name, tag.c_str(), tag.length() ) ){
                if( name[tag.length()] == ':' ) name += tag.length() + 1;
            }
            if( !strcmp( name, "lr") )            base_lr_ = (float)atof(val);
            if( !strcmp( name, "eta") )           base_lr_ = (float)atof(val);
            if( !strcmp( name, "wd") )            wd = (float)atof(val);
            if( !strcmp( name, "momentum") )      momentum = (float)atof(val);
            if( !strcmp( name, "silent") )        silent = atoi(val);

            if( !strncmp( name, "lr:", 3 ) || !strncmp( name, "eta:",4 ) ) {
                if( !strncmp( name, "lr:", 3 ) ) name += 3;
                else name += 4;
                if( !strcmp( name, "schedule") ){
                    if( !strcmp( val, "constant") )  lr_schedule = 0;
                    if( !strcmp( val, "expdecay") )  lr_schedule = 1;
                    if( !strcmp( val, "polydecay") ) lr_schedule = 2;
                    if( !strcmp( val, "factor"))     lr_schedule = 3;
                }
                if( !strcmp( name, "gamma") ) lr_gamma = (float)atof( val );
                if( !strcmp( name, "alpha") ) lr_alpha = (float)atof( val );
                if( !strcmp( name, "step") )  lr_step = atol( val );
                if( !strcmp( name, "factor") ) lr_factor = (float)atof( val );
                if( !strcmp( name, "minimum_lr")) lr_minimum = (float)atof( val );
                if( !strcmp( name, "start_epoch")) start_epoch = atol( val );
            }
        }
    };
}; // namespace cxxnet

namespace cxxnet{
    template<typename xpu, int dim>
    class SGDUpdater : public IUpdater{
    public:
        SGDUpdater( mshadow::Tensor<xpu,dim> &w, mshadow::Tensor<xpu,dim> &dw, const char *tag )
            :w(w), dw(dw){
            param.tag = tag;
            m_w.Resize( w.shape, 0.0f );
        }
        virtual ~SGDUpdater( void ){}
        virtual void Init( void ){
            if( param.silent == 0 ){
                printf("SGDUpdater: eta=%f, mom=%f\n", param.base_lr_, param.momentum );
            }
        }
        virtual void Update( long epoch ){
            param.ScheduleEpoch( epoch );
            m_w *= param.momentum;
            m_w += (-param.learning_rate) * ( dw + param.wd * w );
            w += m_w;
            // dw accumulate gradient instead of storing them, updater need to reset then to 0 after each update
            dw = 0.0f;
        }
        virtual void StartRound( int round ) {
            param.round = round;
        }
        virtual void SetParam( const char *name, const char *val ){
            param.SetParam( name, val );
        }
        virtual void GetData( mshadow::TensorContainer<cpu,2>& weight,
                              mshadow::TensorContainer<cpu,2>& gradient ) const {
            weight.Resize( w.shape.FlatTo2D() );
            gradient.Resize( weight.shape );
            mshadow::Copy( weight, w.FlatTo2D() );
            mshadow::Copy( gradient, dw.FlatTo2D() );
        }
        virtual void SetData( const mshadow::Tensor<cpu,2>& weight,
                              const mshadow::Tensor<cpu,2>& gradient ) {
            mshadow::TensorContainer<xpu,2> tmp_w( weight.shape );
            mshadow::TensorContainer<xpu,2> tmp_dw( gradient.shape );
            mshadow::Copy( tmp_w, weight );
            mshadow::Copy( tmp_dw, gradient );
            w = reshape( tmp_w, w.shape );
            dw = reshape( tmp_dw, dw.shape );
        }
    private:
        UpdaterParam param;
        // variales
        mshadow::Tensor<xpu,dim> &w, &dw;
        // momentum variable
        mshadow::TensorContainer<xpu,dim> m_w;
    };
}; // namespace cxxnet

namespace cxxnet{
    template<typename xpu, int dim>
    class SGHMCUpdater : public IUpdater{
    public:
        SGHMCUpdater( mshadow::Random<xpu> &rnd, mshadow::Tensor<xpu,dim> &w, mshadow::Tensor<xpu,dim> &dw, const char *tag )
            :rnd(rnd),w(w),dw(dw){
            param.tag = tag;
            m_w.Resize( w.shape, 0.0f );
            temp.Resize( w.shape );
        }
        virtual ~SGHMCUpdater( void ){}
        virtual void StartRound( int round ) {
            param.round = round;
            param.hyper_sampled = 0;
        }
        virtual void Init( void ){
            if( param.silent == 0 ){
                printf("SGDHMCUpdater: eta=%f, mom=%f\n", param.base_lr_, param.momentum );
            }
        }
        // update model parameters
        virtual void Update( long epoch ){
            param.ScheduleEpoch( epoch );
            if( param.need_hypersample() && param.hyper_sampled  == 0 ) {
                this->UpdateHyper(); param.hyper_sampled = 1;
            }
            m_w *= param.momentum;
            m_w += (-param.learning_rate) * ( dw + param.wd * w );
            if( param.need_sample() ){
                m_w += rnd.gaussian( w.shape ) * param.GetSigma();
            }
            w += m_w;
            // dw accumulate gradient instead of storing them, updater need to reset then to 0 after each update
            dw = 0.0f;
        }
        // update hyper parameters
        virtual void UpdateHyper( void ){
            mshadow::Copy( temp, w );
            mshadow::Tensor<cpu,2> wt = temp.FlatTo2D();
            double sumcnt = wt.shape[1] * wt.shape[0];
            double sumsqr = 0.0f;
            // note: make the norm sum operation in mshadow
            for( index_t y = 0; y < wt.shape[1]; ++ y )
                for( index_t x = 0; x < wt.shape[0]; ++ x ){
                    sumsqr += wt[y][x] * wt[y][x];
                }
            double alpha = param.hyper_alpha + 0.5 * sumcnt;
            double beta  = param.hyper_beta  + 0.5 * sumsqr;
            double plambda;
            if( param.temp < 1e-6f ){
                plambda = std::max( alpha - 1.0, 0.0 ) / beta;
            }else{
                plambda = utils::SampleGamma( alpha, beta );
            }
            // set weight decay
            param.wd = static_cast<float>( plambda / param.num_train );
            if( param.silent == 0 && param.print_hupdate != 0 ){
                printf("hyperupdate[");
                for( int i = dim-1; i > 0 ; --i ){
                    printf("%u,", temp.shape[i] );
                }
                printf("%u]:plambda=%f,wd=%f\n", temp.shape[0], plambda, param.wd );
            }
        }
        virtual void SetParam( const char *name, const char *val ){
            param.SetParam( name, val );
        }
        virtual void GetData( mshadow::TensorContainer<cpu,2>& weight,
                              mshadow::TensorContainer<cpu,2>& gradient ) const {
            weight.Resize( w.shape.FlatTo2D() );
            gradient.Resize( weight.shape );
            mshadow::Copy( weight, w.FlatTo2D() );
            mshadow::Copy( gradient, dw.FlatTo2D() );
        }
        virtual void SetData( const mshadow::Tensor<cpu,2>& weight,
                              const mshadow::Tensor<cpu,2>& gradient ) {
            mshadow::TensorContainer<xpu,2> tmp_w( weight.shape );
            mshadow::TensorContainer<xpu,2> tmp_dw( gradient.shape );
            mshadow::Copy( tmp_w, weight );
            mshadow::Copy( tmp_dw, gradient );
            w = reshape( tmp_w, w.shape );
            dw = reshape( tmp_dw, dw.shape );
        }
    protected:
        struct HMCParam : public UpdaterParam {
            // when to start sample
            int start_sample;
            // when to start hyper parameter sampling
            int start_hsample;
            // number of training data
            int num_train;
            // Gamma(alpha, beta) prior on regularizer
            float hyper_alpha;
            float hyper_beta;
            // sample hyper parameter each gap_hsample over training data
            int gap_hsample;
            int hyper_sampled;
            // print hyper update
            int print_hupdate;
            // temperature
            float temp;
            // output precision matrix
            float lambda_output;
            // output preiction matrix
            HMCParam( void ){
                start_sample  = INT_MAX;
                start_hsample = INT_MAX;
                temp  = 1.0f;
                hyper_alpha = hyper_beta = 1.0f;
                gap_hsample = 1;
                lambda_output = 1.0f;
                hyper_sampled = 0;
                print_hupdate  = 0;
            }
            inline void SetParam( const char *name, const char* val ) {
                UpdaterParam::SetParam( name, val );
                if( !strncmp( name, tag.c_str(), tag.length() ) ){
                    if( name[tag.length()] == ':' ) name += tag.length() + 1;
                }
                if( !strcmp( "start_sample", name ) )  start_sample = atoi( val );
                if( !strcmp( "start_hsample", name ) ) start_hsample = atoi( val );
                if( !strcmp( "gap_hsample", name ) )   gap_hsample = atoi( val );
                if( !strcmp( "num_train", name ) )     num_train = atoi( val );
                if( !strcmp( "temp", name ) )          temp = (float)atof( val );
                if( !strcmp( "print_hupdate", name ) ) print_hupdate = atoi( val );
                if( !strcmp( "lambda_output", name ) ) lambda_output = (float)atof( val );
            }
            inline bool need_sample( void ) const{
                return round >= start_sample;
            }
            inline bool need_hypersample( void ) const{
                int diff = round - start_hsample;
                return diff >= 0 && diff % gap_hsample == 0;
            }
            inline real_t GetSigma( void ) const{
                real_t scale;
                if ( momentum < 1e-6f ){
                    scale = learning_rate / (num_train * lambda_output);
                }else{
                    scale = learning_rate * (1.0f-momentum) / ( num_train * lambda_output );
                }
                return std::sqrt( 2.0f * temp * scale );
            }
        };
    private:
        // training parameter
        HMCParam param;
        // momentum variable
        mshadow::TensorContainer<xpu,dim> m_w;
        mshadow::TensorContainer<cpu,dim> temp;
        // PRNG
        mshadow::Random<xpu> &rnd;
        mshadow::Tensor<xpu,dim> &w, &dw;
    };
};

namespace cxxnet{
    template<typename xpu, int dim>
    inline IUpdater* CreateUpdater( const char *type,
                                    mshadow::Random<xpu> &rnd,
                                    mshadow::Tensor<xpu,dim> &weight,
                                    mshadow::Tensor<xpu,dim> &wgrad,
                                    const char *tag ){
        if( !strcmp( type, "sgd" ) ) return new SGDUpdater<xpu,dim>( weight, wgrad, tag );
        if( !strcmp( type, "sghmc" ) ) return new SGHMCUpdater<xpu,dim>( rnd, weight, wgrad, tag );
        Error("unknown updater type");
        return NULL;
    }
}; // namespace cxxnet;
#endif // CXXNET_UPDATER_INL_HPP

