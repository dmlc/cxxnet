#ifndef CXXNET_UPDATER_INL_HPP
#define CXXNET_UPDATER_INL_HPP
#pragma once
/*!
 * \file cxxnet_updater-inl.hpp
 * \brief implementation of updaters
 * \author Tianqi Chen
 */
#include <climits>
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
        const char *tag;
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
        /*! \brief constructor that sets default parameters */
        UpdaterParam( void ){
            learning_rate = 0.01f;
            wd = 0.0f;
            momentum = 0.9f;
            silent = 0;
        }
        /*!
         * \brief Set param for the layer from string
         * \param name parameter name
         * \param val string for configuration
         */
        inline void SetParam( const char *name, const char* val ) {
            // if we set "bias:wd = 0.0", and tag == "bias", the it will set wd in current updater param
            // but will not affect settings with other tags
            if( !strncmp( name, tag, strlen(tag) ) ){
                int ltag = strlen(tag);
                if( name[ltag] == ':' ) name += ltag + 1;
            }
            if( !strcmp( name, "lr") )  learning_rate = (float)atof(val);
            if( !strcmp( name, "eta") ) learning_rate = (float)atof(val);
            if( !strcmp( name, "wd") )            wd = (float)atof(val);
            if( !strcmp( name, "momentum") )      momentum = (float)atof(val);
            if( !strcmp( name, "silent") )        silent = atoi(val);
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
        }
        virtual ~SGDUpdater( void ){}
        virtual void Init( void ){
            if( param.silent == 0 ){
                printf("SGDUpdater: eta=%f, mom=%f\n", param.learning_rate, param.momentum );
            }
        }
        virtual void Update( void ){
            dw += param.wd * w;
            w  += (-param.learning_rate) * dw;
            dw *= param.momentum;
        }
        virtual void StartRound( int round ) {
            param.round = round;
        }
        virtual void SetParam( const char *name, const char *val ){
            param.SetParam( name, val );
        }
    private:
        UpdaterParam param;
        mshadow::Tensor<xpu,dim> &w, &dw; 
    };
}; // namespace cxxnet

namespace cxxnet{
    template<typename xpu, int dim>
    class SGHMCUpdater : public IUpdater{
    public:
        SGHMCUpdater( mshadow::Random<xpu> &rnd, mshadow::Tensor<xpu,dim> &w, mshadow::Tensor<xpu,dim> &dw, const char *tag )
            :rnd(rnd), w(w), dw(dw){
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
                printf("SGDHMCUpdater: eta=%f, mom=%f\n", param.learning_rate, param.momentum );
            }
        }
        // update model parameters
        virtual void Update( void ){
            if( param.need_hypersample() && param.hyper_sampled  == 0 ) {
                this->UpdateHyper(); param.hyper_sampled = 1;
            }
            m_w *= param.momentum;
            m_w += (-param.learning_rate) * ( dw + param.wd * w );
            if( param.need_sample() ){
                m_w += rnd.gaussian( w.shape ) * param.GetSigma();
            }
            w += m_w;
            // set dw = 0, so we get fresh gradient 
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
                if( !strncmp( name, tag, strlen(tag) ) ){
                    int ltag = strlen(tag);
                    if( name[ltag] == ':' ) name += ltag + 1;
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
        // weights, gradient accumulater
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
    
