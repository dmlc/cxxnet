#ifndef CXXNET_UPDATER_INL_HPP
#define CXXNET_UPDATER_INL_HPP
#pragma once
/*!
 * \file cxxnet_updater-inl.hpp
 * \brief implementation of updaters
 * \author Tianqi Chen
 */
#include "cxxnet_net.h"

namespace cxxnet{
    // expr is needed to use expression
    using namespace mshadow::expr;
    using namespace mshadow::utils;

    /*! \brief potential parameters for each layer */
    struct UpdaterParam{
        /*! \brief tag of current parameter group */
        const char *tag;
        /*! \brief learning rate */
        float learning_rate;
        /*! \brief weight decay */
        float wd;
        /*! \brief momentum decay */
        float momentum;
        /*! \brief constructor that sets default parameters */
        UpdaterParam( void ){
            learning_rate = 0.01f;
            wd = 0.0f;
            momentum = 1.0f;
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
        virtual void Update( void ){
            dw += param.wd * w;
            w  += (-param.learning_rate) * dw;
            dw *= param.momentum;
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
    inline IUpdater* CreateUpdater( const char *type,
                                    mshadow::Random<xpu> &rnd, 
                                    mshadow::Tensor<xpu,dim> &weight, 
                                    mshadow::Tensor<xpu,dim> &wgrad,
                                    const char *tag ){
        if( !strcmp( type, "sgd" ) ) return new SGDUpdater<xpu,dim>( weight, wgrad, tag );
        Error("unknown updater type");
        return NULL;
    }    
}; // namespace cxxnet;
#endif // CXXNET_UPDATER_INL_HPP
