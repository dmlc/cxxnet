#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <string>
#include <vector>
#include "cxxnet_data.h"

namespace cxxnet{
    IIterator<DataBatch>* CreateIterator( const std::vector< std::pair<std::string,std::string> > &cfg ){
        return NULL;
    }
};
