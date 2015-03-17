#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
// this is where the actual implementations are
#include "nnet_impl-inl.hpp"
// specialize the gpu implementation
namespace cxxnet {
namespace nnet {
template<>
INetTrainer* CreateNet<gpu>(int net_type) {
  return CreateNet_<gpu>(net_type);
}
}  // namespace nnet
}  // namespace cxxnet
