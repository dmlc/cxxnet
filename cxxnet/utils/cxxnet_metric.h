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
             * \param preds prediction
             * \param labels label
             * \param n number of instances
             */
            virtual void AddEval( const float* preds, const float* labels, int n ) = 0;
            /*! \brief get current result */
            virtual double Get( void ) const = 0;
            /*! \return name of metric */
            virtual const char *Name( void ) const= 0;
        };

        /*! \brief RMSE */
        struct MetricRMSE : public IMetric{
        public:
            MetricRMSE( void ){
                this->Clear();
            }
            virtual ~MetricRMSE( void ){}
            virtual void Clear( void ){
                sum_err = 0.0; cnt_inst = 0;
            }
            virtual void AddEval( const float* preds, const float* labels, int ndata ){
                for( int i = 0; i < ndata; ++ i ){
                    float diff = preds[i] - labels[i];
                    sum_err += diff * diff;
                    cnt_inst+= 1;
                }
            }
            virtual double Get( void ) const{
                return std::sqrt( sum_err / cnt_inst );
            }
            virtual const char *Name( void ) const{
                return "rmse";
            }
        private:
            double sum_err;
            long   cnt_inst;
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
            virtual void AddEval( const float* preds, const float* labels, int ndata ){
                for( int i = 0; i < ndata; ++ i ){
                    const float x = preds[i] - 0.5f;
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
        struct MetricError : public IMetric{
        public:
            MetricError( void ){
                this->Clear();
            }
            virtual ~MetricError( void ){}
            virtual void Clear( void ){
                sum_err = 0.0; cnt_inst = 0;
            }
            virtual void AddEval( const float* preds, const float* labels, int ndata ){
                for( int i = 0; i < ndata; ++ i ){
                    sum_err += (int)preds[i] != (int)labels[i];
                    cnt_inst+= 1;
                }
            }

            virtual void AddEval(const float* preds, const float* labels, int ndata, int top_n) {
                bool hit = false;
                for( int i = 0; i < ndata; i += top_n ){
                    hit = false;
                    for (int j = 0; j < top_n; ++j) {
                        if ((int)pred[i + j] == (int)labels[i / top_n]) hit = true;
                    }
                    sum_srr += (int) (!hit);
                    cnt_inst += 1;
                }
            }



            virtual double Get( void ) const{
                return sum_err / cnt_inst;
            }
            virtual const char *Name( void ) const{
                return "error";
            }
        private:
            double sum_err;
            long   cnt_inst;
        };

        /*! \brief a set of evaluators */
        struct MetricSet{
        public:
            void AddMetric( const char *name ){
                if( !strcmp( name, "rmse") ) evals_.push_back( &rmse_ );
                if( !strcmp( name, "error") ) evals_.push_back( &error_ );
                if( !strcmp( name, "r2") )    evals_.push_back( &corrsqr_ );
                // simple way to enforce uniqueness, not a good way, not ok here
                std::sort( evals_.begin(), evals_.end() );
                evals_.resize( std::unique( evals_.begin(), evals_.end() ) - evals_.begin() );
            }
            inline void Clear( void ){
                for( size_t i = 0; i < evals_.size(); ++ i ){
                    evals_[i]->Clear();
                }
            }
            inline void AddEval( const float* preds, const float* labels, int ndata ){
                for( size_t i = 0; i < evals_.size(); ++ i ){
                    evals_[i]->AddEval( preds, labels, ndata );
                }
            }
            inline void Print( FILE *fo, const char *evname ){
                for( size_t i = 0; i < evals_.size(); ++ i ){
                    fprintf( fo, "\t%s-%s:%f", evname, evals_[i]->Name(), evals_[i]->Get() );
                }
            }
        private:
            MetricRMSE  rmse_;
            MetricError error_;
            MetricCorrSqr corrsqr_;
            std::vector<IMetric*> evals_;
        };
    };
};
#endif
