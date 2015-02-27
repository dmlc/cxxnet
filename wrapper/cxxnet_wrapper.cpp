#include <sstream>
#include <string>
#include <mshadow/tensor.h>
#include "./cxxnet_wrapper.h"
#include "../src/utils/config.h"
#include "../src/nnet/nnet.h"
#include "../src/io/data.h"

namespace cxxnet {
class WrapperIterator {
 public:
  WrapperIterator(const char *s_cfg) : iter_(NULL) {
    std::string cstr(s_cfg); cstr += "\n";
    std::stringstream stream(cstr);
    utils::ConfigStreamReader cfg(stream);
    cfg.Init();
    int flag = 1;
    std::vector<std::pair<std::string, std::string> > itcfg;
    std::vector<std::pair<std::string, std::string> > defcfg;

    while (cfg.Next()) {
      const char *name = cfg.name();
      const char *val  = cfg.val();
      if (!strcmp(name, "iter") && !strcmp(val, "end")) {
        utils::Assert(flag != 0, "wrong configuration file");
        iter_  = cxxnet::CreateIterator(itcfg);
        flag = 0; itcfg.clear(); continue;
      }
      if (flag == 0) {
        defcfg.push_back(std::make_pair(std::string(name),
                                        std::string(val)));
      } else {
        itcfg.push_back(std::make_pair(std::string(name),
                                       std::string(val)));
      }
    }
    if (iter_ == NULL) {
      iter_  = cxxnet::CreateIterator(itcfg);
    }
    for (size_t i = 0; i < defcfg.size(); ++i) {
      iter_->SetParam(defcfg[i].first.c_str(),
                      defcfg[i].second.c_str());
    }
    iter_->Init();
  }
  ~WrapperIterator(void) {
    delete iter_;
  }

 private:
  friend class WrapperNet;
  IIterator<DataBatch> *iter_;
};

class WrapperNet {
 public:
  WrapperNet(const char *device, const char *s_cfg)
      : net_(NULL) {
    device = "gpu";
    net_type = 0;
    silent = 0;
    print_step = 100;
    this->Configure(s_cfg);
    if (device != NULL && device[0] != '\0') {
      this->SetParam("dev", device);
    }
  }
  ~WrapperNet(void) {
    delete net_;
  }
  inline void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "dev")) device_type_ = val;
    if (!strcmp(name,"net_type") && net_ != NULL) {
      net_type = atoi(val); return;
    }
    if (!strcmp(name, "silent")) {
      silent = atoi(val); return;
    }
    if (!strcmp(name, "print_step")) {
      print_step = atoi(val); return;
    }
    if (net_ != NULL) net_->SetParam(name, val);
    cfg.push_back(std::make_pair(std::string(name),
                                 std::string(val)));
  }
  inline void InitModel(void) {
    if (net_ != NULL) delete net_;
    net_ = this->CreateNet();
    net_->InitModel();
  }
  // load model from file
  inline void LoadModel(const char *fname) {
    if (net_ != NULL) delete net_;
    FILE *fi = utils::FopenCheck(fname, "rb");
    utils::FileStream fs(fi);
    utils::Check(fs.Read(&net_type, sizeof(int)) != 0, "LoadModel");
    net_ = this->CreateNet();
    net_->LoadModel(fs);
    fclose(fi);
  }
  // save model into file
  inline void SaveModel(const char *fname) {
    FILE *fo  = utils::FopenCheck(fname, "wb");
    utils::FileStream fs(fo);
    fs.Write(&net_type, sizeof(int));
    net_->SaveModel(fs);
    fclose(fo);
  }
  inline void StartRound(int round) {
    round_counter = round;
  }
  inline void UpdateOneIter(WrapperIterator *iter) {
    IIterator<DataBatch> *itr_train = iter->iter_;
    int sample_counter = 0;
    itr_train->BeforeFirst();
    while (itr_train->Next()) {
      net_->Update(itr_train->Value());
      if (++sample_counter % print_step == 0) {
        if (silent == 0) {
          printf("\r                                                               \r");
          printf("round %8d:[%8d] ", round_counter, sample_counter);
          fflush(stdout);
        }
      }
    }
  }
  inline cxx_real_t *Predict(const DataBatch &batch, cxx_uint *out_size) {
    res_pred = 0.0f;
    net_->Predict(&res_pred, batch);
    *out_size = static_cast<cxx_uint>(res_pred.size(0));
    return &res_pred[0];
  }
  inline cxx_real_t *PredictIter(WrapperIterator *iter, cxx_uint *out_size) {
    res_pred_all.clear();
    IIterator<DataBatch> *itr_data = iter->iter_;
    itr_data->BeforeFirst();
    while (itr_data->Next()) {
      res_pred = 0.0f;
      net_->Predict(&res_pred, itr_data->Value());
      *out_size += static_cast<cxx_uint>(res_pred.size(0));
      for (cxx_uint i = 0; i < res_pred.size(0); ++i) {
        res_pred_all.push_back(res_pred[i]);
      }
    }
    return BeginPtr(res_pred_all);
  }
  inline const char *Evaluate(WrapperIterator *iter, const char *data_name) {
    res_eval = net_->Evaluate(iter->iter_, data_name);
    return res_eval.c_str();
  }
  // return the net
  inline nnet::INetTrainer *net(void) {
    return net_;
  }

 protected:
  // returning cache
  std::string res_eval;
  mshadow::TensorContainer<mshadow::cpu, 1> res_pred;
  std::vector<cxx_real_t> res_pred_all;
 private:
  // the internal net
  nnet::INetTrainer *net_;
  /*! \brief all the configurations */
  std::vector<std::pair< std::string, std::string> > cfg;
  /*! \brief  device of the trainer */
  std::string device_type_;
  /*! \brief type of net implementation */
  int net_type;
  // silence sign
  int silent;
  // print step
  int print_step;
  // rounter counter
  int round_counter;

  inline void Configure(const char *s_cfg) {
    std::string cstr(s_cfg); cstr += "\n";
    std::stringstream sstream(cstr);
    utils::ConfigStreamReader cfg(sstream);
    cfg.Init();
    while (cfg.Next()) {
      this->SetParam(cfg.name(), cfg.val());
    }
  }
  // create a neural net
  inline nnet::INetTrainer *CreateNet(void) {
    nnet::INetTrainer *net;
    if (!strncmp(device_type_.c_str(), "gpu", 3)) {
#if MSHADOW_USE_CUDA
      net = nnet::CreateNet<mshadow::gpu>(net_type);
#else
      utils::Error("MSHADOW_USE_CUDA was not enabled");
#endif
    } else {
      net = nnet::CreateNet<mshadow::cpu>(net_type);
    }
    for (size_t i = 0; i < cfg.size(); ++ i) {
      net->SetParam(cfg[i].first.c_str(), cfg[i].second.c_str());
    }
    return net;
  }
};
}  // namespace cxxnet

using namespace cxxnet;

extern "C" {
  void *CXNIOCreateFromConfig(const char *cfg) {
    return new WrapperIterator(cfg);
  }
  void CXNIOFree(void *handle) {
    delete static_cast<WrapperIterator*>(handle);
  }
  void *CXNNetCreate(const char *device, const char *cfg) {
    return new WrapperNet(device, cfg);
  }
  void CXNNetFree(void *handle) {
    delete static_cast<WrapperNet*>(handle);
  }
  void CXNNetSetParam(void *handle, const char *name, const char *val) {
    static_cast<WrapperNet*>(handle)->SetParam(name, val);
  }
  void CXNNetInitModel(void *handle) {
    static_cast<WrapperNet*>(handle)->InitModel();
  }
  void CXNNetSaveModel(void *handle, const char *fname) {
    static_cast<WrapperNet*>(handle)->SaveModel(fname);
  }
  void CXNNetLoadModel(void *handle, const char *fname) {
    static_cast<WrapperNet*>(handle)->LoadModel(fname);
  }
  void CXNNetStartRound(void *handle, int round) {
    static_cast<WrapperNet*>(handle)->StartRound(round);
  }
  void CXNNetUpdateOneIter(void *handle, void *data_handle) {
    static_cast<WrapperNet*>(handle)->
        UpdateOneIter(static_cast<WrapperIterator*>(data_handle));
  }
  void CXNNetUpdateOneBatch(void *handle,
                            cxx_real_t *p_data,
                            cxx_uint nbatch,
                            cxx_uint nchannel,
                            cxx_uint height,
                            cxx_uint width,
                            cxx_real_t *p_label,
                            cxx_uint label_width) {
    DataBatch batch;
    batch.label = mshadow::Tensor<cpu, 2>
        (p_label, mshadow::Shape2(nbatch, label_width));
    batch.batch_size = nbatch;
    batch.data = mshadow::Tensor<cpu, 4>
        (p_data, mshadow::Shape4(nbatch, nchannel, height, width));
    static_cast<WrapperNet*>(handle)->net()->Update(batch);
  }
  const cxx_real_t *CXNNetPredict(void *handle,
                                  cxx_real_t *p_data,
                                  cxx_uint nbatch,
                                  cxx_uint nchannel,
                                  cxx_uint height,
                                  cxx_uint width,
                                  cxx_uint *out_size) {
    DataBatch batch;
    batch.batch_size = nbatch;
    batch.data = mshadow::Tensor<cpu, 4>
        (p_data, mshadow::Shape4(nbatch, nchannel, height, width));
    return static_cast<WrapperNet*>(handle)->Predict(batch, out_size);
  }
  const cxx_real_t *CXNNetPredictIter(void *handle,
                                       void *data_handle,
                                       cxx_uint *out_size) {
    WrapperIterator* iter = static_cast<WrapperIterator*>(data_handle);
    return static_cast<WrapperNet*>(handle)->PredictIter(iter, out_size);
  }
  const char *CXNNetEvaluate(void *handle,
                             void *data_handle,
                             const char *data_name) {
    return static_cast<WrapperNet*>(handle)->
        Evaluate(static_cast<WrapperIterator*>(data_handle), data_name);
  }
}
