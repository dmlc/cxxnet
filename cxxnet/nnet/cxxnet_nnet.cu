#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#include "cxxnet_nnet-inl.hpp"

namespace cxxnet {
    template<typename xpu>
    INetTrainer* CreateNet_( int net_type ){
        switch( net_type ){
        case 0: return new CXXNetTrainer<xpu>();
        case 1: return new CXXAvgNetTrainer<xpu>();
        default: Error("unknown net type");
        }
        return NULL;
    }
    INetTrainer* CreateNet( int net_type, const char *device ){
        if( !strcmp( device, "gpu") ) return CreateNet_<gpu>( net_type );
        if( !strcmp( device, "cpu") ) return CreateNet_<cpu>( net_type );
        Error("unknown device type" );
        return NULL;
    }
};
