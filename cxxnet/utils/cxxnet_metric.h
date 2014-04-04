#ifndef CXXNET_METRIC_H
#define CXXNET_METRIC_H
#pragma once
/*!
 * \file cxxnet_metric.h
 * \brief evaluation metrics
 * \author Tianqi Chen
 */
#include <cmath>
#include <vector>
#include <algorithm>
#include "cxxnet_global_random.h"

namespace cxxnet{
    namespace utils{
        /*! \brief evaluator that evaluates the loss metrics */
        class IMetric{
        public:
            IMetric( void ){}
            /*!\brief virtual destructor */
            virtual ~IMetric( void ){}
            /*! \brief clear statistics */
            virtual void Clear( void ) = 0;
            /*!
             * \brief evaluate a specific metric, add to current statistics
             * \param preds prediction score array
             * \param labels label
             * \param n number of instances
             */
            virtual void AddEval( const mshadow::Tensor<cpu,2> &predscore, const float* labels ) = 0;
            /*! \brief get current result */
            virtual double Get( void ) const = 0;
            /*! \return name of metric */
            virtual const char *Name( void ) const= 0;
        };

        /*! \brief simple metric Base */
        struct MetricBase : public IMetric{
        public:
            virtual ~MetricBase( void ){}
            virtual void Clear( void ){
                sum_metric = 0.0; cnt_inst = 0;
            }
            virtual void AddEval( const mshadow::Tensor<cpu,2> &predscore, const float* labels ) {
                for( index_t i = 0; i < predscore.shape[1]; ++ i ){                    
                    sum_metric += CalcMetric( predscore[i], labels[i] );
                    cnt_inst+= 1;
                }
            }
            virtual double Get( void ) const{
                return sum_metric / cnt_inst;
            }
            virtual const char *Name( void ) const{
                return name.c_str();
            }
        protected:
            MetricBase( const char *name ){
                this->name = name;
                this->Clear();
            }
            virtual float CalcMetric( const mshadow::Tensor<cpu,1> &predscore, float label ) = 0;
        private:
            double sum_metric;
            long   cnt_inst;
            std::string name;
        };

        /*! \brief RMSE */
        struct MetricRMSE : public MetricBase{
        public:
            MetricRMSE( void ):MetricBase( "rmse" ){
            }
            virtual ~MetricRMSE( void ){}
        protected:
            virtual float CalcMetric( const mshadow::Tensor<cpu,1> &predscore, float label ) {
                utils::Assert( predscore.shape[0] == 1,"RMSE can only accept shape[0]=1" );
                float diff = predscore[0] - label;
                return diff*diff;
            }
        };

        /*! \brief r^2 correlation square */
        struct MetricCorrSqr : public IMetric{
        public:
            MetricCorrSqr( void ){
                this->Clear();
            }
            virtual ~MetricCorrSqr( void ){}
            virtual void Clear( void ){
                sum_x = 0.0; sum_y = 0.0;
                sum_xsqr  = 0.0;
                sum_ysqr  = 0.0;
                sum_xyprod = 0.0;
                cnt_inst = 0;
            }
            virtual void AddEval( const mshadow::Tensor<cpu,2> &predscore, const float* labels ) {
                utils::Assert( predscore.shape[0] == 1,"RMSE can only accept shape[0]=1" );
                for( index_t i = 0; i < predscore.shape[1]; ++ i ){                    
                    const float x = predscore[i][0] - 0.5f;
                    const float y = labels[i] - 0.5f;
                    sum_x += x; sum_y += y;
                    sum_xsqr += x * x;
                    sum_ysqr += y * y;
                    sum_xyprod += x * y;
                    cnt_inst += 1;
                }
            }
            virtual double Get( void ) const{
                double mean_x = sum_x / cnt_inst;
                double mean_y = sum_y / cnt_inst;
                double corr = sum_xyprod / cnt_inst - mean_x*mean_y;
                double xvar = sum_xsqr / cnt_inst  - mean_x*mean_x;
                double yvar = sum_ysqr / cnt_inst  - mean_y*mean_y;
                double res =  corr * corr / ( xvar * yvar );

                return res;
            }
            virtual const char *Name( void ) const{
                return "r2";
            }
        private:
            inline static float sqr( float x ){
                return x*x;
            }
            double sum_x, sum_y;
            double sum_xsqr, sum_ysqr;
            double sum_xyprod;
            long   cnt_inst;
        };

        /*! \brief Error */
        struct MetricError : public MetricBase{
        public:
            MetricError( void ):MetricBase("error"){
            }
            virtual ~MetricError( void ){}
        protected:
            virtual float CalcMetric( const mshadow::Tensor<cpu,1> &pred, float label ) {
                index_t maxidx = 0;
                for( index_t i = 1; i < pred.shape[0]; ++ i ){
                    if( pred[i] > pred[maxidx] ) maxidx = i;
                }
                return maxidx !=(index_t)label;
            }
        };


        /*! \brief Recall@n */
        struct MetricRecall : public MetricBase{
        public:
            MetricRecall( const char *name ): MetricBase(name){
                utils::Assert( sscanf( name, "rec@%d", &topn) == 1, "must specify n for rec@n" );
            }
            virtual ~MetricRecall( void ){}
        protected:
            virtual float CalcMetric( const mshadow::Tensor<cpu,1> &pred, float label ) {
                if( pred.shape[0] < (index_t)topn ){
                    fprintf( stderr, "evaluating rec@%d, list=%u", topn, pred.shape[0] );
                    utils::Error("it is meaningless to take rec@n for list shorter than n" );                    
                }
                index_t klabel = (index_t)label;
                vec.resize( pred.shape[0] );
                for( index_t i = 0; i < pred.shape[0]; ++ i ){
                    vec[i] = std::make_pair( pred[i], i );
                }
                Shuffle( vec );
                std::sort( vec.begin(), vec.end(), CmpScore );
                for( int i = 0; i < topn; ++ i ) {
                    if( vec[i].second == klabel ) return 1.0f;
                }
                return 0.0f;
            }
        private:
            inline static bool CmpScore( const std::pair<float,index_t> &a, const std::pair<float,index_t> &b ){
                return a.first > b.first;
            }
            std::vector< std::pair<float,index_t> > vec;
            int topn;
        };

        /*! \brief a set of evaluators */
        struct MetricSet{
        public:
            ~MetricSet( void ){
                for( size_t i = 0; i < evals_.size(); ++ i ){
                    delete evals_[i];
                }
            }
            void AddMetric( const char *name ){
                if( !strcmp( name, "rmse") )  evals_.push_back( new MetricRMSE() );
                if( !strcmp( name, "error") ) evals_.push_back( new MetricError() );
                if( !strcmp( name, "r2") )    evals_.push_back( new MetricCorrSqr() );
                if( !strncmp( name, "rec@",4) )  evals_.push_back( new MetricRecall( name ) );
                // simple way to enforce uniqueness, not a good way, not ok here
                std::sort( evals_.begin(), evals_.end(), CmpName );
                evals_.resize( std::unique( evals_.begin(), evals_.end(), EqualName ) - evals_.begin() );
            }
            inline void Clear( void ){
                for( size_t i = 0; i < evals_.size(); ++ i ){
                    evals_[i]->Clear();
                }
            }
            inline void AddEval( const mshadow::Tensor<cpu,2> &predscore, const float* labels ) {
                for( size_t i = 0; i < evals_.size(); ++ i ){
                    evals_[i]->AddEval( predscore, labels );
                }
            }
            inline void Print( FILE *fo, const char *evname ){
                for( size_t i = 0; i < evals_.size(); ++ i ){
                    fprintf( fo, "\t%s-%s:%f", evname, evals_[i]->Name(), evals_[i]->Get() );
                }
            }
        private:
            inline static bool CmpName( const IMetric *a, const IMetric *b ){
                return strcmp( a->Name(), b->Name() ) < 0;
            }
            inline static bool EqualName( const IMetric *a, const IMetric *b ){
                return strcmp( a->Name(), b->Name() ) == 0;
            }
        private:
            std::vector<IMetric*> evals_;
        };
    };
};
#endif
