#include <cstdio>
#include "cxxnet/cxxnet.h"
using namespace cxxnet;

int main( void ){
    Node<cpu> in, out;
    in.data = mshadow::NewCTensor( mshadow::Shape4(1,1,4,2) , 1.0f );
    out.data = in.data;
    FullConnectLayer<cpu> layer( in, out);
    return 0;
}
