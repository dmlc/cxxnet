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
            std::vector<IMetric*> evals_;  
        };
    };
};
#endif
