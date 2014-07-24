#ifndef CXXNET_DATA_H
#define CXXNET_DATA_H
#pragma once
/*!
 * \file cxxnet_data.h
 * \brief data type and iterator abstraction
 * \author Bing Xu, Tianqi Chen
 */

#include <vector>
#include <string>
#include "mshadow/tensor.h"
#include "../utils/cxxnet_utils.h"

namespace cxxnet {
    /*! 
     * \brief iterator type 
     * \tparam DType data type
     */
    template<typename DType>
    class IIterator{
    public:
        /*!
         * \brief set the parameter 
         * \param name name of parameter
         * \param val  value of parameter
         */        
        virtual void SetParam( const char *name, const char *val ) = 0;
        /*! \brief initalize the iterator so that we can use the iterator */
        virtual void Init( void ) = 0;       
        /*! \brief set before first of the item */
        virtual void BeforeFirst( void ) = 0;
        /*! \brief move to next item */
        virtual bool Next( void ) = 0;
        /*! \brief get current data */
        virtual const DType &Value( void ) const = 0;
    public:
        /*! \brief constructor */
        virtual ~IIterator( void ){}
    };
    
    /*! \brief a single data instance */
    struct DataInst{
        /*! \brief label information */
        float  label;
        /*! \brief unique id for instance */
        unsigned index;
        /*! \brief content of data */
        mshadow::Tensor<mshadow::cpu,3> data;
    };

    /*! \brief a sparse data instance, in sparse vector */
    struct SparseInst{ 
        /*! \brief an entry of sparse vector */
        struct Entry{
            /*! \brief feature index */
            unsigned findex;
            /*! \brief feature value */
            float fvalue;
        };
        /*! \brief label information */
        float  label;
        /*! \brief unique id for instance */
        unsigned index; 
        /*! \brief length of the instance */
        unsigned length;
        /*! \brief pointer to the elements*/
        const Entry *data;
        /*! \brief get i-th pair in the sparse vector*/
        inline const Entry& operator[](size_t i)const{
            return data[i];
        }
    };

    /*! 
     * \brief a standard batch of data commonly used by iterator 
     *        this could be a dense batch or sparse matrix, 
     *        the iterator configuration should be aware what kind of batch can be generated and feed in to each type of neuralnet
     */
    struct DataBatch{
    public:
        /*! \brief label information */
        float*  labels;
        /*! \brief unique id for instance, can be NULL, sometimes is useful */
        unsigned* inst_index;
        /*! \brief number of instance */
        mshadow::index_t batch_size;
        /*! \brief number of padding elements in this batch, 
             this is used to indicate the last elements in the batch are only padded up to match the batch, and should be discarded */
        mshadow::index_t num_batch_padd;
    public:
        /*! \brief content of dense data, if this DataBatch is dense */
        mshadow::Tensor<mshadow::cpu,4> data;
    public:
        // sparse part of the DataBatch, in CSR format
        /*! \brief array[batch_size+1], row pointer of each of the elements */
        size_t *sparse_row_ptr;
        /*! \brief array[row_ptr.back()], content of the sparse element */
        SparseInst::Entry *sparse_data;
    public:        
        /*! \brief constructor */
        DataBatch( void ){
            labels = NULL; inst_index = NULL; 
            data.dptr = NULL; 
            batch_size = 0; num_batch_padd = 0;
            sparse_row_ptr = NULL;
            sparse_data = NULL;
        }
        /*! \brief auxiliary  functionto allocate space, if needed */
        inline void AllocSpaceDense( mshadow::Shape<4> shape, mshadow::index_t batch_size, bool pad = false ){
            data = mshadow::NewTensor<mshadow::cpu>( shape, 0.0f, pad );
            labels = new float[ batch_size ];
            inst_index = new unsigned[ batch_size ];
            this->batch_size = batch_size;
        }
        /*! \brief auxiliary function to free space, if needed, dense only */
        inline void FreeSpaceDense( void ){
            if( labels != NULL ){
                delete [] labels;
                delete [] inst_index;
                mshadow::FreeSpace( data );
                labels = NULL;
            }
        }
        /*! \brief copy dense content from existing data, dense only */
        inline void CopyFromDense( const DataBatch &src ){
            utils::Assert( batch_size == src.batch_size );
            memcpy( labels, src.labels, batch_size * sizeof( float ) );
            memcpy( inst_index, src.inst_index, batch_size * sizeof( unsigned ) );
            mshadow::Copy( data, src.data );
        }
    public:
        /*! \brief helper function to check if a element is sparse */
        inline bool is_sparse(void) const{
            return sparse_row_ptr != NULL;
        }
        /*! \brief get rid'th row from the sparse element, the data must be in sparse format */
        inline SparseInst GetRowSparse(unsigned rid) const{
            SparseInst inst;
            inst.data = sparse_data + sparse_row_ptr[rid];
            inst.length = sparse_row_ptr[rid+1]- sparse_row_ptr[rid];
            inst.label = labels[rid];
            if( inst_index != NULL ){
                inst.index = inst_index[rid];
            }else{
                inst.index = 0; 
            }
            return inst;
        }
    };
};

namespace cxxnet {
    /*! 
     * \brief create iterator from configure settings 
     * \param cfg configure settings key=vale pair
     */
    IIterator<DataBatch>* CreateIterator( const std::vector< std::pair<std::string,std::string> > &cfg );
};
#endif
