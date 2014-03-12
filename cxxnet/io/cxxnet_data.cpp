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
#include "../utils/cxxnet_io_utils.h"
#include "cxxnet_iter_mnist-inl.hpp"

namespace cxxnet{
    IIterator<DataBatch>* CreateIterator( const std::vector< std::pair<std::string,std::string> > &cfg ){
        IIterator<DataBatch>* it = new MNISTIterator();
        for( size_t i = 0; i < cfg.size(); ++ i ){
            it->SetParam( cfg[i].first.c_str(), cfg[i].second.c_str() );
        }
        it->Init();
        return it;
    }
};
