#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <map>
#include <sstream>
#include <mshadow-ps/mshadow_ps.h>
#include "./nnet_config.h"
#include "../layer/param.h"
#include "../utils/config.h"
#include "../updater/updater.h"

#if MSHADOW_DIST_PS
namespace PS {
DECLARE_string(app_file);
} // namespace PS
#endif

namespace cxxnet {
namespace nnet {
class CXXNetUpdater : public mshadow::ps::IModelUpdater<real_t> {
 public:
  CXXNetUpdater(void) : rnd(0) {
    seed = 0;
  }
  virtual ~CXXNetUpdater(void) {
    for (std::map<int, UpdaterEntry*>::iterator
             it = updaters.begin(); it != updaters.end(); ++it) {
      delete it->second;
    }
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "seed")) seed = atoi(val);
    cfgvec.push_back(std::make_pair(std::string(name), std::string(val)));
  }

  virtual void InitUpdater(int rank, int argc, char *argv[]) {
    if (argc < 2) {
      printf("Usage: <config>\n");
      exit(0);
    }

    utils::ConfigIterator itr(argv[1]);
    while (itr.Next()) {
      this->SetParam(itr.name(), itr.val());
    }
    for (int i = 2; i < argc; i ++) {
      char name[256], val[256];
      if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
        this->SetParam(name, val);
      }
    }

    // start configure settings
    cfg.Configure(cfgvec);
    rnd.Seed(seed + rank * 17);
  }
  virtual void InitModel(int key, real_t *dptr, size_t size) {
    if (updaters.find(key) != updaters.end()) {
      // already inited
      // TODO do some checks here
      return;
    }
    updaters[key] = new UpdaterEntry();
    UpdaterEntry &e = *updaters[key];
    e.key = key;
    e.weight = mshadow::Tensor<cpu, 1>
        (dptr, mshadow::Shape1(size)).FlatTo2D();
    e.updater = updater::CreateUpdater<cpu>
        (cfg.updater_type.c_str(),
         &rnd, e.weight, e.weight,
         updater::DecodeTag(key));
    e.is_bias = !strcmp(updater::DecodeTag(key), "bias");
    const int i = key / updater::kDataKeyStep;
    utils::Assert(i < cfg.param.num_layers, "layer index exceed bound");
    e.layer_type = cfg.layers[i].type;
    for (size_t j = 0; j < cfg.defcfg.size(); ++j) {
      e.SetParam(cfg.defcfg[j].first.c_str(),
                 cfg.defcfg[j].second.c_str());
    }
    for (size_t j = 0; j < cfg.layercfg[i].size(); ++j) {
      e.SetParam(cfg.layercfg[i][j].first.c_str(),
                 cfg.layercfg[i][j].second.c_str());
    }
    e.Init(&rnd);
  }
  virtual void Update(int key, real_t *dptr, size_t size) {
    std::map<int, UpdaterEntry*>::iterator it
        = updaters.find(key);
    utils::Assert(it != updaters.end() && it->first == key,
                  "must call initkey first before calling update");
    it->second->Update(dptr, size);
  }

 private:
  struct UpdaterEntry {
    int key;
    // whether this is bias
    bool is_bias;
    // type of layer
    layer::LayerType layer_type;
    // epoch we run
    long epoch;
    // parameters
    layer::LayerParam param;
    updater::IUpdater<cpu> *updater;
    mshadow::Tensor<cpu, 2> weight;
    // constructor
    UpdaterEntry(void) : epoch(0) {
      updater = NULL;
    }
    ~UpdaterEntry(void) {
      delete updater;
    }
    inline void SetParam(const char *name,
                         const char *val) {
      updater->SetParam(name, val);
      param.SetParam(name, val);
    }
    inline void Init(mshadow::Random<cpu> *p_rnd) {
      updater->Init();
      if (is_bias) {
        weight = param.init_bias;
      } else {
        if (layer_type == layer::kConv) {
          param.RandInitWeight(p_rnd, weight, param.num_channel, param.kernel_height *param.kernel_width);
        } else {
          utils::Check(param.random_type != 1 || param.init_uniform > 0.0f,
                       "xavier not supported in PS");
          param.RandInitWeight(p_rnd, weight, param.num_hidden, param.num_hidden);
        }
      }
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

 private:
  int seed;
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

namespace mshadow {
namespace ps {
template<>
IModelUpdater<cxxnet::real_t> *CreateModelUpdater<cxxnet::real_t>(void) {
  return new cxxnet::nnet::CXXNetUpdater();
}
}  // namespace ps
}  // namespace mshadow

#if MSHADOW_DIST_PS
int CreateServerNode(int argc, char *argv[]) {
  mshadow::ps::MShadowServerNode<float> server(argc, argv);
  return 0;
}

#endif
