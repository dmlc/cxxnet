#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// GPU version
#include "cxxnet_nnet-inl.hpp"

namespace cxxnet {
    INetTrainer* CreateNetGPU( int net_type ){
        return CreateNet_<gpu>( net_type );
    }
};
