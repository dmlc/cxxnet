#ifndef CXXNET_CORE_H
#define CXXNET_CORE_H
#pragma once
/*!
 * \file cxxnet_core.h
 * \brief Abstruct definition for layer interface,
 *        data type, everything used to construct a network
 * \author Tianqi Chen, Bing Xu
 */

/*! \brief whether to adapt caffe layers */
#ifndef CXXNET_ADAPT_CAFFE
#define CXXNET_ADAPT_CAFFE 0
#endif

#include <vector>
#include <limits>
#include <queue>
#include <string>
#include "mshadow/tensor.h"
#include "mshadow/tensor_io.h"
#include "../utils/cxxnet_utils.h"

/*! \brief namespace of cxxnet */
namespace cxxnet {
    typedef mshadow::cpu cpu;
    typedef mshadow::gpu gpu;
    typedef mshadow::index_t index_t;
    typedef mshadow::real_t  real_t;
};


namespace cxxnet {
    /*! \brief interface of a updater */
    class IUpdater{
    public:
        /*!\brief virtual destructor */
        virtual ~IUpdater( void ){}
        /*! \brief intialize, print information about updater if not silent */
        virtual void Init( void ) = 0;
        /*! \brief update parameter */
        virtual void Update( void ) = 0;        
        /*!
         * \brief inform the updater that starting round iteration over data
         * \param round round counter
         */
        virtual void StartRound( int round ) = 0;
        /*!\ brief set parameters that could be spefic to this updater */
        virtual void SetParam( const char *name, const char *val ) = 0;
    };

    /*! \brief interface of layer */
    class ILayer {
    public:
        /*!\brief virtual destructor */
        virtual ~ILayer( void ){}
        /*!
         * \brief Forward propagation from in_node to out_node
         * \param is_train the propagation is training or dropout
         */
        virtual void Forward(bool is_train) = 0;
        /*!
         * \brief Back propagation from out_node to in_node, generate the gradient, out_node already stores gradient value
         * \param prop_grad if true, then the layer will propagate gradient back to its input node
         */
        virtual void Backprop(bool prop_grad) = 0;
    public:
        // interface code that not needed to be implemented by all nodes
        /*!
         * \brief adjust output node shape, according to current layers' configuration
         */
        virtual void AdjustNodeShape( void ){}
        /*!
         * \brief Get updaters for the layer
         * \param specified updater type
         * \param updaters the laeyer will push_back into updaters
         */
        virtual void GetUpdaters( const char *updater, std::vector<IUpdater*> &updaters ) {}
        /*!
         * \brief Set param for the layer from string
         * \param name parameter name
         * \param val string for configuration
         */
        virtual void SetParam(const char *name, const char* val) {}
        /*!
         * \brief intialized model parameters
         */
        virtual void InitModel(void) {}
        /*!
         * \brief Save model into binary file
         * \param fo output stream
         */
        virtual void SaveModel(mshadow::utils::IStream &fo) const {}
        /*!
         * \brief Load model from binary file
         * \param fi input stream
         */
        virtual void LoadModel(mshadow::utils::IStream &fi) {}
    };
}; // namespace cxxnet

// data structures
namespace cxxnet {
    template<typename xpu>
    struct Node;

    /*! \brief abstruct class for Node */
    template<typename xpu>
    class NodeFactory{
    public:
        NodeFactory( void ){
            if( xpu::kDevCPU ){
                // CPU, always not use backup
                max_mem_ = std::numeric_limits<size_t>::max();
            }else{
                // GPU, 1.6 GB bydefault
                max_mem_ = 400L * 1000000;
            }
            warning_ = 1;
            total_mem_ = 0;
        }
        /* create new node */
        inline Node<xpu> CreateNode( void ){
            return Node<xpu>( this );
        }
        /* set memory limits in terms of MB */
        inline void SetMemLimit( const char *size ){
            if( !xpu::kDevCPU ){
                float n;
                if( sscanf( size, "%fMB", &n ) == 1 ){
                    this->max_mem_ = static_cast<size_t>( n * (1L<<20) / 4 ); return;
                }
                if( sscanf( size, "%fGB", &n ) == 1 ){
                    this->max_mem_ = static_cast<size_t>( n * (1L<<30) / 4 ); return;
                }
                warning_ = 1;
                mshadow::utils::Error("unknown memory limit string");
            }
        }
    private:
        friend class Node<xpu>;
        /*! \brief request memory */
        inline void ReqMem( mshadow::Tensor<xpu,4> &data ){
            size_t mem = data.shape.MSize();
            total_mem_ += mem;
            if( total_mem_ > max_mem_ && warning_ != 0 ){
                printf("warning: hit total memory limit, start swap mode\n");
                warning_ = 0;
            }
            while( total_mem_ > max_mem_ ){
                utils::Assert( !free_list_.empty(), "can not meet memory requirement" );
                Node<xpu> *n = free_list_.front(); free_list_.pop();
                utils::Assert( n->data.dptr != NULL, "BUG" );
                if( !n->pinned_ ) {
                    if( n->backup_.dptr == NULL ){
                        n->backup_.shape = n->data.shape;
                        mshadow::AllocSpace( n->backup_ );
                    }
                    mshadow::Copy( n->backup_, n->data );
                    mshadow::FreeSpace( n->data );
                    n->data.dptr = NULL;
                }
                n->inqueue_ = false;
            }
            mshadow::AllocSpace( data );
            total_mem_ += data.shape.MSize() - mem;
        }
        /*! \brief register the node as unpinned */
        inline void RegUnpin( Node<xpu> *n ){
            if( n->inqueue_ == true ) return;
            n->inqueue_ = true;
            free_list_.push( n );
        }
    private:
        /*! \brief whether do warning when memory swap occurs */
        int warning_;
        /*! \brief maximum memory allowed in total for nodes */
        size_t max_mem_;
        /*! \brief total amount of memory */
        size_t total_mem_;
        std::queue< Node<xpu>* > free_list_;
    }; // class NodeFactory

    template<typename xpu>
    struct Node {
    public:
        /*! \brief content of the node */
        mshadow::Tensor<xpu,4> data;
    public:
        /*! \brief free space of node */
        inline void FreeSpace( void ){
            if( backup_.dptr != NULL ) mshadow::FreeSpace( backup_ );
            if( data.dptr != NULL )   mshadow::FreeSpace( data );
        }
            /*! \brief matrix view of the node */
        inline mshadow::Tensor<xpu,2> mat( void ){
            return data[0][0];
        }
            /*! \brief whether it holds a matrix data */
        inline bool is_mat( void ) const{
            return data.shape[2] == 1 && data.shape[3] == 1;
        }
        /*! \brief pin the data into xpu memory,  will ensure it is there */
        inline void Pin( void ){
            if( data.dptr != NULL ) return;
            pinned_ = true;
            parent_->ReqMem( data );
            if( backup_.dptr != NULL ){
                mshadow::Copy( data, backup_ );
            }
        }
        /*! \brief unpin the data, data can be deallocated */
        inline void Unpin( void ){
            if( !pinned_ ) return;
            pinned_ = false;
            parent_->RegUnpin( this );
        }
    public:
        /* public constructor, use with caution */
        Node( void ){
            data.dptr = NULL;
            backup_.dptr = NULL;
        }
    private:
        /*! \brief allow factory to see node */
        friend class NodeFactory<xpu>;
        /*! \brief constructor */
        Node( NodeFactory<xpu>* parent ):parent_(parent){
            pinned_ = false;
            inqueue_ = false;
            data.dptr = NULL;
            backup_.dptr = NULL;
        }
        /*! \brief whether data is pinned */
        bool pinned_;
        /*! \brief whether data is in queue */
        bool inqueue_;
        /*! \brief pointer to parent */
        NodeFactory<xpu> *parent_;
        /*! \brief backup content of the node */
        mshadow::Tensor<cpu,4> backup_;
    }; // struct Node
}; // namespace cxxnet

namespace cxxnet {
    /*!
     * \brief factory: create an upadater algorithm of given type
     * \param type indicate the type of updater
     * \param rnd random number generator
     * \param weight network weight
     * \param grad network gradient
     * \param tag some tags used to identify the weight, for example: "bias", "wmat", "mask", default ""
     */
    template<typename xpu, int dim>
    inline IUpdater* CreateUpdater( const char *type,
                                    mshadow::Random<xpu> &rnd,
                                    mshadow::Tensor<xpu,dim> &weight,
                                    mshadow::Tensor<xpu,dim> &wgrad,
                                    const char *tag );
}; // namespace cxxnet

namespace cxxnet {
    /*! \brief namespace for type of layer */
    namespace layer_type{
        const int kFullConnect = 0;
        const int kSoftmax     = 1;
        const int kRectifiedLinear = 2;
        const int kSigmoid = 3;
        const int kTanh = 4;
        const int kSoftplus = 5;
        const int kFlatten  = 6;
        const int kDropout = 7;
        const int kDropConn = 8;
        const int kConv = 9;
        const int kCaffe = 100;
    };
    /*! \brief namespace for type of random init method */
    namespace rnd_type {
        const int kGaussian = 0;
        const int kUniform = 1;
    };
    /*!
     * \brief factory: create an upadater algorithm of given type
     * \param type indicate the type of a layer
     * \param rnd random number generator
     * \param in input node
     * \param out output node
     */
    template<typename xpu>
    inline ILayer* CreateLayer( const char *type, mshadow::Random<xpu> &rnd, Node<xpu>& in, Node<xpu>& out );
    /*!
     * \brief factory: create an upadater algorithm of given type
     * \param type indicate the type of a layer
     */
    inline int GetLayerType( const char *type );
};  // namespace cxxnet

#include "cxxnet_updater-inl.hpp"
#include "cxxnet_layer-inl.hpp"

#endif // CXXNET_NET_H
