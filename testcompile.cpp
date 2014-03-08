#include <cstdio>
#include "cxxnet/cxxnet.h"
using namespace cxxnet;

int main( void ){
    mshadow::Random<cpu> rnd(1);
    Node<cpu> in, out;
    in.data = mshadow::NewCTensor( mshadow::Shape4(1,1,4,2) , 1.0f );
    out.data = in.data;
    FullConnectLayer<cpu> layer( rnd, in, out);
    ILayer *l = CreateLayer( "fullc", rnd, in, out );
    delete l;
    return 0;
}
