#ifndef CXXNET_ITER_BATCH_INL_HPP
#define CXXNET_ITER_BATCH_INL_HPP
#pragma once
/*!
 * \file cxxnet_iter_batch-inl.hpp
 * \brief create a batch iterator from single instance iterator
 * \author Tianqi Chen
 */
#include "mshadow/tensor.h"
#include "mshadow/tensor_container.h"
#include "cxxnet_data.h"

namespace cxxnet {
    class BatchAdaptIterator: public IIterator<DataBatch>{
    public:
        BatchAdaptIterator( IIterator<DataInst> *base ):base_(base){}
        virtual ~BatchAdaptIterator( void ){
            delete base_;
            out_.FreeSpace();
        }
        virtual void SetParam( const char *name, const char *val ){
            base_->SetParam( name, val );
            if( !strcmp( name, "batch_size") )  shape_[3] = (index_t)atoi( val );
            if( !strcmp( name, "input_shape") ) {
                utils::Assert( sscanf( val, "%u,%u,%u", &shape_[2],&shape_[1],&shape_[0] ) ==3,
                               "input_shape must be three consecutive integers without space example: 1,1,200 " );
            }
        }
        virtual void Init( void ){
            base_->Init();
            out_.AllocSpace( shape_, shape_[3], false );
        }
        virtual void BeforeFirst( void ){
            base_->BeforeFirst();
        }
        virtual bool Next( void ){
            using namespace mshadow::expr;
            index_t top = 0;
            while( base_->Next() ){
                const DataInst &d = base_->Value();
                out_.labels[top] = d.label;
                out_.inst_index[top] = d.index;
                out_.data[top] = crop( d.data, out_.data[0][0].shape );
                if( ++ top >= shape_[3] ) return true;
            }
            return false;
        }
        virtual const DataBatch &Value( void ) const{
            return out_;
        }
    private:
        // input shape
        mshadow::Shape<4> shape_;
        // base iterator
        IIterator<DataInst> *base_;
        // output data
        DataBatch out_;        
    };
};
#endif
