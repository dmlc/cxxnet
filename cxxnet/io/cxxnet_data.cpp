#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <string>
#include <vector>
#include "mshadow/tensor.h"

namespace cxxnet {
    typedef mshadow::cpu cpu;
    typedef mshadow::gpu gpu;
    typedef mshadow::index_t index_t;
    typedef mshadow::real_t  real_t;
};

#include "cxxnet_data.h"
#include "../utils/cxxnet_utils.h"
#include "../utils/cxxnet_io_utils.h"
#include "cxxnet_iter_cifar-inl.hpp"
#include "cxxnet_iter_mnist-inl.hpp"
#include "cxxnet_iter_spfeat-inl.hpp"
#include "cxxnet_iter_img-inl.hpp"
#include "cxxnet_iter_proc-inl.hpp"
#include "cxxnet_iter_thread_imbin-inl.hpp"

namespace cxxnet{
    IIterator<DataBatch>* CreateIterator( const std::vector< std::pair<std::string,std::string> > &cfg ){
        size_t i = 0;
        IIterator<DataBatch>* it = NULL;
        for(; i < cfg.size(); ++i ){
            const char* name = cfg[i].first.c_str();
            const char* val  = cfg[i].second.c_str();
            if( !strcmp( name, "iter" ) ){
                if( !strcmp( val, "mnist") ){
                    utils::Assert( it == NULL );
                    it = new MNISTIterator(); continue;
                }
                if( !strcmp( val, "spfeat") ){
                    utils::Assert( it == NULL );
                    it = new SpFeatIterator(); continue;
                }
                if( !strcmp( val, "cifar") ) {
                    utils::Assert( it == NULL );
                    it = new CIFARIterator(); continue;
                }
                if( !strcmp( val, "image") ) {
                    utils::Assert( it == NULL );
                    it = new BatchAdaptIterator( new ImageIterator() ); continue;
                }
                if( !strcmp( val, "imgbin")) {
                     utils::Assert( it == NULL );
                     it = new BatchAdaptIterator(new ThreadImagePageIterator()); continue;
                }
                if( !strcmp( val, "threadbuffer") ){
                    utils::Assert( it != NULL, "must specify input of threadbuffer" );
                    it = new ThreadBufferIterator( it );
                    continue;
                }
                utils::Error("unknown iterator type" );
            }

            if( it != NULL ){
                it->SetParam( name, val );
            }
        }
        utils::Assert( it != NULL, "must specify iterator by iter=itername" );
        return it;
    }
};
