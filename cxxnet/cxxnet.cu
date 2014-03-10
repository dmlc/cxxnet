#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "cxxnet.h"
#include "cxxnet_net.h"
#include "cxxnet_net-inl.hpp"

namespace cxxnet {
    template<typename xpu>
    INetTrainer* CreateNet_( int net_type ){
        return new CXXNetTrainer<xpu>();
    }
    INetTrainer* CreateNet( int net_type, const char *device ){
        if( !strcmp( device, "gpu") ) return CreateNet_<gpu>( net_type );
        else{
            return CreateNet_<cpu>( net_type );
        }
    }
};
