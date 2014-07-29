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
    
    /*! \brief adapter to adapt instance iterator into batch iterator */
    class SparseBatchAdapter: public IIterator<DataBatch>{
    public:
        SparseBatchAdapter( IIterator<SparseInst> *base ):base_(base){
            batch_size_ = 0;
        }   
        virtual ~SparseBatchAdapter(void){
            delete base_;
        }
        virtual void SetParam( const char *name, const char *val ){
            base_->SetParam( name, val );
            if( !strcmp(name, "round_batch") ) round_batch_ = atoi(val);
            if( !strcmp(name, "batch_size") ) batch_size_ = (unsigned)atoi(val);
        }
        virtual void Init( void ){
            base_->Init();
            labels.resize( batch_size_ );
            inst_index.resize( batch_size_ );
            out_.labels = &labels[0];
            out_.inst_index = &inst_index[0];           
        }
        virtual void BeforeFirst( void ){
            if( round_batch_ == 0 || num_overflow_ == 0 ){
                // otherise, we already called before first
                base_->BeforeFirst();
            }else{
                num_overflow_ = 0;
            }
        }
        virtual bool Next( void ){
            data.clear();
            row_ptr.clear(); row_ptr.push_back(0);
            out_.num_batch_padd = 0;
            out_.batch_size = batch_size_;

            while(base_->Next()){
                this->Add( base_->Value() );
                if( row_ptr.size() > batch_size_ ){
                    out_.sparse_row_ptr = &row_ptr[0]; 
                    out_.sparse_data = &data[0];
                    return true;
                }
            }
            if( row_ptr.size() == 1 ) return false;
            out_.num_batch_padd = batch_size_ + 1 - row_ptr.size();
            if( round_batch_ != 0 ){
                num_overflow_ = 0;
                base_->BeforeFirst();
                for( ;row_ptr.size() <= batch_size_;  ++num_overflow_ ){
                    utils::Assert( base_->Next(), "number of input must be bigger than batch size" );
                    this->Add( base_->Value() );
                }
            }else{
                while( row_ptr.size() <= batch_size_ ){
                    row_ptr.push_back( row_ptr.back() );
                }                    
            }
            utils::Assert( row_ptr.size() == batch_size_+1, "BUG" );
            out_.sparse_row_ptr = &row_ptr[0]; 
            out_.sparse_data = &data[0];
            return true;
        }
        virtual const DataBatch &Value( void ) const{
            return out_;
        }        
    private:
        inline void Add( const SparseInst &line ){
            labels[row_ptr.size()-1] = line.label;
            inst_index[row_ptr.size()-1] = line.index;
            for(unsigned i = 0; i < line.length; ++ i){
                data.push_back( line[i] );
            }
            row_ptr.push_back( row_ptr.back() + line.length );
        }
    private:
        // base iterator
        IIterator<SparseInst> *base_;
        // batch size
        index_t batch_size_;
        // use round batch mode to reuse further things
        int round_batch_;
        // number of overflow items 
        int num_overflow_;
    private:
        // output data        
        DataBatch out_;
        // actual content of row ptr
        std::vector<size_t> row_ptr;
        std::vector<float>  labels;
        std::vector<unsigned> inst_index;
        // actual content of sparse data storage
        std::vector<SparseInst::Entry> data;
    };
};
#endif
