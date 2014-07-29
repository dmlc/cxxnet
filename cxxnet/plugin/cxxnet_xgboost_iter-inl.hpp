#ifndef CXXNET_XGBOOST_ITER_INL_HPP
#define CXXNET_XGBOOST_ITER_INL_HPP
#pragma once
/*!
 * \file cxxnet_xgboost_iter-inl.hpp
 * \brief adapt input format of xgboost DMatrix format into cxxnet
 * \author Tianqi Chen
 */

#include <string>
#include <cstring>
#include "../io/cxxnet_data.h"
#include "../utils/cxxnet_global_random.h"
#include "regrank/xgboost_regrank_data.h"

namespace cxxnet{
    /*! \brief iterator adapter of xgboost */
    class XGBoostIterator:  public IIterator<SparseInst>{
    public:
        XGBoostIterator(void){
            shuffle_ = 0;
            silent_ = 0;
            index_offset_ = 0;
        }
        virtual ~XGBoostIterator(void){}
        virtual void SetParam( const char *name, const char *val ) {
            if( !strcmp(name, "path_data") ) fname_ = val;
            if( !strcmp(name, "silent") ) silent_ = atoi(val);
            if( !strcmp(name, "shuffle") ) shuffle_ = atoi(val);           
        }
        virtual void Init( void ) {
            dmat_.CacheLoad( fname_.c_str(), false, false );
            index_set_.resize( dmat_.Size() );
            for(size_t i = 0; i < index_set_.size(); ++i){
                index_set_[i] = (unsigned)i;
            }
            if( shuffle_ ) utils::Shuffle( index_set_ );
            if( silent_ == 0 ){
                printf("XGBoostIterator: data=%s, %lu data loaded, shuffle=%d\n", fname_.c_str(), index_set_.size(), shuffle_ );
            } 
        }
        virtual void BeforeFirst( void ) {
            row_index_ = 0;
        }        
        virtual bool Next( void ) {
            if( row_index_ >= index_set_.size() ) return false;
            entry_.clear();
            unsigned ridx = index_set_[row_index_];
            for( xgboost::booster::FMatrixS::RowIter it = dmat_.data.GetRow(ridx); it.Next();){
                entry_.push_back( SparseInst::Entry(it.findex(), it.fvalue()) );
            }
            out_.data = &entry_[0];
            out_.index =  ridx + index_offset_;
            out_.length = (unsigned)entry_.size();
            if( dmat_.info.labels.size() != 0 ) out_.label = dmat_.info.labels[ridx];
            ++ row_index_;
            return true;
        }
        virtual const SparseInst &Value( void ) const {
            return out_;
        }
    private:
        SparseInst out_;
        std::vector<SparseInst::Entry> entry_;
    private:
        int silent_;
        int shuffle_;
        size_t row_index_;
        std::string fname_;
        unsigned index_offset_;
    private:
        std::vector<unsigned> index_set_;
        xgboost::regrank::DMatrix dmat_;
    };
};
#endif
