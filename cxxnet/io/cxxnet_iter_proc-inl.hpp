#ifndef CXXNET_ITER_PROC_INL_HPP
#define CXXNET_ITER_PROC_INL_HPP
#pragma once
/*!
 * \file cxxnet_iter_proc-inl.hpp
 * \brief definition of preprocessing iterators that takes an iterator and do some preprocessing 
 * \author Tianqi Chen
 */
#include "mshadow/tensor.h"
#include "mshadow/tensor_container.h"
#include "cxxnet_data.h"
#include "../utils/cxxnet_thread_buffer.h"

namespace cxxnet {
    /*! \brief create a batch iterator from single instance iterator */
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
            index_t batch_size = shape_[3];
            if( shape_[2] == 1 && shape_[1] == 1 ){
                shape_[1] = shape_[3]; shape_[3] = 1;
            }
            out_.AllocSpace( shape_, batch_size, false );
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
        // base iterator
        IIterator<DataInst> *base_;
        // batch size
        index_t batch_size_;
        // input shape
        mshadow::Shape<4> shape_;
        // output data
        DataBatch out_;        
    };
};

namespace cxxnet{
    /*! \brief thread buffer iterator */
    class ThreadBufferIterator: public IIterator< DataBatch >{
    public :
        ThreadBufferIterator( IIterator<DataBatch> *base ){
            silent_ = 0;
            itr.get_factory().base_ = base;
            itr.SetParam( "buffer_size", "32" );
        }
        virtual ~ThreadBufferIterator(){
            itr.Destroy();
        }
        virtual void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "silent") ) silent_ = atoi( val );
            itr.SetParam( name, val );
        }        
        virtual void Init( void ){
            utils::Assert( itr.Init() ) ;
            if( silent_ == 0 ){
                printf( "ThreadBufferIterator\n" );
            }            
        }
        virtual void BeforeFirst(){
            itr.BeforeFirst();
        }
        virtual bool Next(){
            if( itr.Next( out_ ) ){
                return true;
            }else{
                return false;
            }
        }       
        virtual const DataBatch &Value() const{
            return out_;
        }
    private:
        struct Factory{
        public:
            IIterator< DataBatch> *base_;
        public:
            Factory( void ){ 
                base_ = NULL; 
            }
            inline void SetParam( const char *name, const char *val ){
                base_->SetParam( name, val );
            }       
            inline bool Init(){ 
                base_->Init();
                utils::Assert( base_->Next(), "ThreadBufferIterator: input can not be empty" );
                oshape_ = base_->Value().data.shape;
                batch_size_ = base_->Value().batch_size;
                base_->BeforeFirst();
                return true;
            }            
            inline bool LoadNext( DataBatch &val ){        
                if( base_->Next() ){
                    val.CopyFrom( base_->Value() );
                    return true;
                }else{
                    return false;
                }
            }
            inline DataBatch Create( void ){
                DataBatch a; a.AllocSpace( oshape_, batch_size_ );
                return a;
            }            
            inline void FreeSpace( DataBatch &a ){        
                a.FreeSpace();
            }                
            inline void Destroy(){
                if( base_ != NULL ) delete base_;
            }            
            inline void BeforeFirst(){
                base_->BeforeFirst();
            }
        private:
            mshadow::index_t batch_size_;
            mshadow::Shape<4> oshape_;
        };        
    private:
        int silent_;
        DataBatch out_;
        utils::ThreadBuffer<DataBatch,Factory> itr;
    };
};
#endif
