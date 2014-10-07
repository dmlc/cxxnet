#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// this is where the actual implementations are
#include "nnet_impl-inl.hpp"
// specialize the cpu implementation
namespace cxxnet {
namespace nnet {
template<>
INetTrainer* CreateNet<cpu>(int net_type) {
  return CreateNet_<cpu>(net_type);
}
}  // namespace nnet
}  // namespace cxxnet
