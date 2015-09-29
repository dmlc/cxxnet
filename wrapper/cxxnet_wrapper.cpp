#include <sstream>
#include <string>
#include <mshadow/tensor.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
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
        CHECK(flag != 0) << "wrong configuration file";
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
  inline void BeforeFirst() {
    iter_->BeforeFirst();
  }
  inline bool Next() {
    return iter_->Next();
  }
  inline const cxx_real_t *GetData(cxx_uint dshape[4], cxx_uint *p_stride) const {
    const DataBatch &batch = iter_->Value();
    for (index_t i = 0; i < 4; ++i) {
      dshape[i] = batch.data.size(i);
    }
    *p_stride = batch.data.stride_;
    return batch.data.dptr_;
  }
  inline const cxx_real_t *GetLabel(cxx_uint lshape[4], cxx_uint *p_stride) const {
    const DataBatch &batch = iter_->Value();
    for (index_t i = 0; i < 2; ++i) {
      lshape[i] = batch.label.size(i);
    }
    *p_stride = batch.label.stride_;
    return batch.label.dptr_;
  }

 private:
  friend class WrapperNet;
  IIterator<DataBatch> *iter_;
};

class WrapperNet {
 public:
  WrapperNet(const char *device, const char *s_cfg)
      : res_pred(false), temp2(false),
        temp4(false), net_(NULL) {
    device = "cpu";
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
    dmlc::Stream *fs = dmlc::Stream::Create(fname, "r");
    utils::Check(fs->Read(&net_type, sizeof(int)) != 0, "LoadModel");
    net_ = this->CreateNet();
    net_->LoadModel(*fs);
    delete fs;
  }
  // save model into file
  inline void SaveModel(const char *fname) {
    dmlc::Stream *fo = dmlc::Stream::Create(fname, "w");
    fo->Write(&net_type, sizeof(int));
    net_->SaveModel(*fo);
    delete fo;
  }
  inline void StartRound(int round) {
    round_counter = round;
  }
  inline cxx_real_t *GetWeight(const char *layer_name,
                               const char *wtag,
                               cxx_uint wshape[4],
                               cxx_uint *out_dim) {
    std::vector<index_t> shape;
    net_->GetWeight(&temp2, &shape, layer_name, wtag);
    *out_dim = static_cast<cxx_uint>(shape.size());
    if (shape.size() == 0) return NULL;
    utils::Check(shape.size() <= 4, "GetWeight only works for dim<=4");
    for (size_t i = 0; i < shape.size(); ++i) {
      wshape[i] = shape[i];
    }
    return temp2.dptr_;
  }
  inline cxx_real_t *Extract(const DataBatch &batch,
                             const char *node_name,
                             cxx_uint oshape[4]) {
    net_->ExtractFeature(&temp4, batch, node_name);
    for (int i = 0; i < 4; ++i) {
      oshape[i] = temp4.size(i);
    }
    return temp4.dptr_;
  }
  inline cxx_real_t *Extract(WrapperIterator *iter,
                             const char *node_name,
                             cxx_uint oshape[4]) {
    return this->Extract(iter->iter_->Value(), node_name, oshape);
  }
  inline void UpdateIter(WrapperIterator *iter) {
    net_->Update(iter->iter_->Value());
  }
  inline cxx_real_t *Predict(const DataBatch &batch, cxx_uint *out_size) {
    net_->Predict(&res_pred, batch);
    *out_size = static_cast<cxx_uint>(res_pred.size(0));
    return res_pred.dptr_;
  }
  inline cxx_real_t *PredictIter(WrapperIterator *iter, cxx_uint *out_size) {
    return Predict(iter->iter_->Value(), out_size);
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
  mshadow::TensorContainer<mshadow::cpu, 2> temp2;
  mshadow::TensorContainer<mshadow::cpu, 4> temp4;
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
  int CXNIONext(void *handle) {
    return static_cast<WrapperIterator*>(handle)->Next();
  }
  void CXNIOBeforeFirst(void *handle) {
    static_cast<WrapperIterator*>(handle)->BeforeFirst();
  }
  const cxx_real_t *CXNIOGetData(void *handle,
                             cxx_uint oshape[4],
                             cxx_uint *ostride) {
    return static_cast<WrapperIterator*>(handle)->GetData(oshape, ostride);
  }
  const cxx_real_t *CXNIOGetLabel(void *handle,
                              cxx_uint oshape[2],
                              cxx_uint *ostride) {
    return static_cast<WrapperIterator*>(handle)->GetLabel(oshape, ostride);
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
  void CXNNetSetWeight(void *handle,
                       cxx_real_t *p_weight,
                       cxx_uint size_weight,
                       const char *layer_name,
                       const char *wtag) {
    mshadow::Tensor<cpu, 2> weight(p_weight, mshadow::Shape2(1, size_weight));
    static_cast<WrapperNet*>(handle)->net()->SetWeight(weight, layer_name, wtag);
  }
  const cxx_real_t *CXNNetGetWeight(void *handle,
                                    const char *layer_name,
                                    const char *wtag,
                                    cxx_uint wshape[4],
                                    cxx_uint *out_dim) {
    return static_cast<WrapperNet*>(handle)->GetWeight(layer_name, wtag, wshape, out_dim);
  }
  void CXNNetUpdateIter(void *handle, void *data_handle) {
    static_cast<WrapperNet*>(handle)->
        UpdateIter(static_cast<WrapperIterator*>(data_handle));
  }
  void CXNNetUpdateBatch(void *handle,
                         cxx_real_t *p_data,
                         const cxx_uint dshape[4],
                         cxx_real_t *p_label,
                         const cxx_uint lshape[2]) {
    DataBatch batch;
    batch.label = mshadow::Tensor<cpu, 2>
        (p_label, mshadow::Shape2(lshape[0], lshape[1]));
    batch.batch_size = dshape[0];
    batch.data = mshadow::Tensor<cpu, 4>
        (p_data, mshadow::Shape4(dshape[0], dshape[1], dshape[2], dshape[3]));
    static_cast<WrapperNet*>(handle)->net()->Update(batch);
  }
  const cxx_real_t *CXNNetPredictBatch(void *handle,
                                       cxx_real_t *p_data,
                                       const cxx_uint dshape[4],
                                       cxx_uint *out_size) {
    DataBatch batch;
    batch.batch_size = dshape[0];
    batch.data = mshadow::Tensor<cpu, 4>
        (p_data, mshadow::Shape4(dshape[0], dshape[1], dshape[2], dshape[3]));
    return static_cast<WrapperNet*>(handle)->Predict(batch, out_size);
  }
  const cxx_real_t *CXNNetPredictIter(void *handle,
                                       void *data_handle,
                                       cxx_uint *out_size) {
    WrapperIterator* iter = static_cast<WrapperIterator*>(data_handle);
    return static_cast<WrapperNet*>(handle)->PredictIter(iter, out_size);
  }
  const cxx_real_t *CXNNetExtractBatch(void *handle,
                                       cxx_real_t *p_data,
                                       const cxx_uint dshape[4],
                                       const char *node_name,
                                       cxx_uint oshape[4]) {
    DataBatch batch;
    batch.batch_size = dshape[0];
    batch.data = mshadow::Tensor<cpu, 4>
        (p_data, mshadow::Shape4(dshape[0], dshape[1], dshape[2], dshape[3]));
    return static_cast<WrapperNet*>(handle)->Extract(batch, node_name, oshape);
  }
  const cxx_real_t *CXNNetExtractIter(void *handle,
                                      void *data_handle,
                                      const char *node_name,
                                      cxx_uint oshape[4]) {
    return static_cast<WrapperNet*>(handle)->Extract
        (static_cast<WrapperIterator*>(data_handle), node_name, oshape);
  }
  const char *CXNNetEvaluate(void *handle,
                             void *data_handle,
                             const char *data_name) {
    return static_cast<WrapperNet*>(handle)->
        Evaluate(static_cast<WrapperIterator*>(data_handle), data_name);
  }
}
