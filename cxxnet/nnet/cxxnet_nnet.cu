#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include "cxxnet_nnet-inl.hpp"

namespace cxxnet {
    INetTrainer* CreateNet( int net_type, const char *device ){
        if( !strcmp( device, "gpu") ) return CreateNet_<gpu>( net_type );
        if( !strcmp( device, "cpu") ) return CreateNet_<cpu>( net_type );
        Error("unknown device type" );
        return NULL;
    }
};
