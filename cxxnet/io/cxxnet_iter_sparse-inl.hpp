#ifndef CXXNET_ITER_SPARSE_INL_HPP
#define CXXNET_ITER_SPARSE_INL_HPP
#pragma once
/*!
 * \file cxxnet_iter_sparse-inl.hpp
 * \brief definition of iterator that relates to sparse data format
 * \author Tianqi Chen
 */

namespace cxxnet {
    /*! \brief a simple adapter */
    class Dense2SparseAdapter: public IIterator<DataBatch>{
    public:
        Dense2SparseAdapter( IIterator<DataBatch> *base ):base_(base){
        }   
        virtual ~Dense2SparseAdapter(void){
            delete base_;
        }
        virtual void SetParam( const char *name, const char *val ){
            base_->SetParam( name, val );
        }
        virtual void Init( void ){
            base_->Init();
        }
        virtual void BeforeFirst( void ){
            base_->BeforeFirst();
        }
        virtual bool Next( void ){
            if(base_->Next()){
                // convert the space
                const DataBatch &din = base_->Value();
                out_.labels = din.labels;
                out_.inst_index = din.inst_index;
                out_.batch_size = din.batch_size;
                out_.num_batch_padd = din.num_batch_padd;
                row_ptr.clear(); row_ptr.push_back(0); data.clear();
                
                utils::Assert( din.batch_size == din.data.shape[1], "Dense2SparseAdapter: only support 1D input");                
                for( mshadow::index_t i = 0; i < din.batch_size; ++i ){
                    mshadow::Tensor<mshadow::cpu, 1> row = din.data[0][0][i];                   
                    for( mshadow::index_t j = 0; j < row.shape[0]; ++j ){
                        data.push_back( SparseInst::Entry( j, row[j] ) );
                    }
                    row_ptr.push_back( row_ptr.back() + row.shape[0] );
                }
                out_.sparse_row_ptr = &row_ptr[0];
                out_.sparse_data = &data[0];
                return true;
            }else{
                return false;
            }
        }
        virtual const DataBatch &Value( void ) const{
            return out_;
        }        
    private:
        // base iterator
        IIterator<DataBatch> *base_;
        // output data        
        DataBatch out_;
        // actual content of row ptr
        std::vector<size_t> row_ptr;
        // actual content of sparse data storage
        std::vector<SparseInst::Entry> data;
    };
};
#endif

