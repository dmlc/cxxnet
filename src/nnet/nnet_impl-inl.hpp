#include "./nnet.h"
#include "./neural_net-inl.hpp"

namespace cxxnet {
namespace nnet {
template<typename xpu>
INetTrainer *CreateNet_(int net_type) {
  NetConfig cfg;
  NeuralNetThread<xpu> thread(cfg, true);
  thread.Update(0);
  return NULL;
}
}  // namespace nnet
}  // cxxnet
