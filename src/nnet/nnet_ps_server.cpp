#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <map>
#include "./nnet_config.h"
#include "../updater/updater.h"

namespace cxxnet {
namespace nnet {
class NetServer {
 public:
  NetServer(void) : rnd(0) {
  }
  ~NetServer(void) {
    for (std::map<int, UpdaterEntry*>::iterator
             it = updaters.begin(); it != updaters.end(); ++it) {
      delete it->second;
    }
  }
  virtual void SetParam(const char *name, const char *val) {
    cfgvec.push_back(std::make_pair(std::string(name), std::string(val)));
  }
  virtual void InitModel(void) {
    cfg.Configure(cfgvec);
    // todo
    rnd.Seed(0);
  }

  virtual void InitKey(int key, real_t *dptr, size_t size) {
    updaters[key] = new UpdaterEntry();
    UpdaterEntry &e = *updaters[key];
    e.key = key;
    e.weight = mshadow::Tensor<cpu, 1>
        (dptr, mshadow::Shape1(size)).FlatTo2D();
    e.updater = updater::CreateUpdater<cpu>
        (cfg.updater_type.c_str(),
         &rnd, e.weight, e.weight,
         updater::DecodeTag(key));
    const int i = key / updater::kDataKeyStep;
    for (size_t j = 0; j < cfg.defcfg.size(); ++j) {
      e.SetParam(cfg.defcfg[j].first.c_str(),
                 cfg.defcfg[j].second.c_str());
    }
    for (size_t j = 0; j < cfg.layercfg[i].size(); ++j) {
      e.SetParam(cfg.layercfg[i][j].first.c_str(),
                 cfg.layercfg[i][j].second.c_str());
    }
    e.Init();
  }
  
 protected:
  struct UpdaterEntry {
    int key;
    long epoch;
    updater::IUpdater<cpu> *updater;
    mshadow::Tensor<cpu, 2> weight;
    UpdaterEntry(void) : epoch(0) {
      updater = NULL;
    }
    ~UpdaterEntry(void) {
      delete updater;
    }
    inline void SetParam(const char *name,
                         const char *val) {
      updater->SetParam(name, val);
    }
    inline void Init(void) {
      // TODO
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
    updaters[key]->Update(grad, size);
  }
  
 private:
  mshadow::Random<cpu> rnd;
  // updaters
  std::map<int, UpdaterEntry*> updaters;
  /*! \brief network configuration type */
  NetConfig cfg;
  /*! \brief history of configurations */
  std::vector< std::pair<std::string, std::string> > cfgvec;
};
}  // namespace nnet
}  // namespace cxxnet
