#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include "./nnet_config.h"
#include "../layer/visitor.h"
#include "./neural_net-inl.hpp"

namespace cxxnet {
namespace nnet {
class NetServer {
 public:
  NetServer(void) : net(NULL) {
    this->SetParam("force_contiguous", "1");
  }
  virtual void SetParam(const char *name, const char *val) {
    cfgvec.push_back(std::make_pair(std::string(name), std::string(val)));
  }
  virtual void InitModel(void) {
    cfg.Configure(cfgvec);
    net = new NeuralNet<cpu>(cfg, batch_size, 0, NULL);
    net->InitModel();
    this->InitUpdaters();
  }

 protected:
  struct UpdaterEntry {
    int key;
    long epoch;
    updater::IUpdater<cpu> *updater;
    mshadow::Tensor<cpu, 2> weight;
    UpdaterEntry() : epoch(0) {      
    }
    // update given gradient
    inline void Update(real_t *grad, size_t size) {
      utils::Assert(size == weight.MSize(),
                    "PS: weight and gradient size inconsistent");
      updater->Update(epoch,
                      mshadow::Tensor<cpu, 2>(grad, weight.shape_));
      epoch += 1;
    }
  };
  // register weights to ps
  virtual void Register(int key, real_t *dptr, size_t size) {
  }
  // callback to handle update
  virtual void Update(int key, real_t *grad, size_t size) {
    utils::Assert(static_cast<size_t>(key) < updaters.size(),
                  "key exceed bound");
    updaters[key].Update(grad, size);
  }
  inline void InitUpdaters(void) {
    for (int i = 0; i < cfg.param.num_layers; ++i) {
      layer::Connection<cpu> &c = net->connections[i];
      std::vector<updater::IUpdater<cpu>*> out;
      if (c.type != layer::kSharedLayer) {
        updater::CreateUpdaters(cfg.updater_type.c_str(),
                                &net->rnd, c.layer, &out);
        for (size_t k = 0; k < out.size(); ++k) {
          for (size_t j = 0; j < cfg.defcfg.size(); ++j) {
            out[k]->SetParam(cfg.defcfg[j].first.c_str(),
                             cfg.defcfg[j].second.c_str());
          }
          for (size_t j = 0; j < cfg.layercfg[i].size(); ++j) {
            out[k]->SetParam(cfg.layercfg[i][j].first.c_str(),
                             cfg.layercfg[i][j].second.c_str());
          }
          out[k]->Init();
          layer::GetWeightVisitor<cpu> vs("weight");
          out[k]->ApplyVisitor(&vs);
          utils::Assert(vs.data.size() == 1,
                        "updater must get exactly one weight");
          UpdaterEntry e;
          e.key = static_cast<int>(updaters.size());
          e.updater = out[k];
          e.weight = vs.data[0];
          utils::Assert(e.weight.CheckContiguous(),
                        "Layer weight do not implement force_contiguous");
          updaters.push_back(e);
          this->Register(e.key, e.weight.dptr_, e.weight.MSize());
        }
      }
    }
  }
  
 private:
  // underlying net code
  NeuralNet<cpu> *net;
  // updaters
  std::vector<UpdaterEntry> updaters;
  // batch size
  mshadow::index_t batch_size;
  /*! \brief serialized model in CPU */
  std::string model_blob;
  /*! \brief network configuration type */
  NetConfig cfg;
  /*! \brief history of configurations */
  std::vector< std::pair<std::string, std::string> > cfgvec;
};

}  // namespace nnet
}  // namespace cxxnet

