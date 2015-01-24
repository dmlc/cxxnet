#include <utility>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include "./nnet.h"
#include "../utils/io.h"
#include "../utils/metric.h"
#include "./neural_net-inl.hpp"


namespace cxxnet {
namespace nnet {
/*! \brief implementation of neural network trainer, using multiple threads */
template<typename xpu>
class CXXNetThreadTrainer : public INetTrainer {
 public:
  CXXNetThreadTrainer(void) {
    batch_size = 100;
    update_period = 1;
    sample_counter = 0;
    eval_train = 1;
    epoch_counter = 0;
    seed = 0;
    pserver = NULL;
    type_pserver = "UNSPECIFIED";
  }
  virtual ~CXXNetThreadTrainer(void) {
    this->FreeNet();
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "dev")) {
      devices_.clear();
      const char *devs = strchr(val, ':');
      if (devs != NULL) {
        int a, b;
        if (sscanf(devs+1, "%d-%d", &a, &b) == 2) {
          for (int i = a; i < b; ++i) {
            devices_.push_back(i);
          }
        } else {
          std::string s_dev = devs + 1;
          char *ptr = strtok(&s_dev[0], ",");
          while (ptr != NULL) {
            utils::Check(sscanf(ptr, "%d", &a) == 1, "invalid device format");
            devices_.push_back(a);
            ptr = strtok(NULL, ",");
          }
        }
      }
    }
    if (!strcmp(name, "batch_size")) batch_size = static_cast<mshadow::index_t>(atoi(val));
    if (!strcmp(name, "update_period")) update_period = atoi(val);
    if (!strcmp(name, "eval_train")) eval_train = atoi(val);
    if (!strcmp(name, "seed")) seed = atoi(val);
    if (!strcmp(name, "param_server")) type_pserver = val;
    if(!strcmp(name, "metric")) {
      metric.AddMetric(val); train_metric.AddMetric(val);
    }
    cfg.push_back(std::make_pair(std::string(name), std::string(val)));
  }
  virtual void InitModel(void) {
    this->InitNet();
    nets_[0]->InitModel();
    nets_[0]->WaitJob();
    this->Save2ModelBlob();
    for(size_t i = 1; i < nets_.size(); ++i) {
      utils::MemoryBufferStream fs(&model_blob_);
      nets_[i]->LoadModel(fs);
      nets_[i]->WaitJob();
    }
    this->InitTemp();
  }
  virtual void SaveModel(utils::IStream &fo) {
    this->Save2ModelBlob();
    net_cfg.SaveNet(fo);
    fo.Write(&epoch_counter, sizeof(epoch_counter));
    fo.Write(model_blob_);
  }
  virtual void LoadModel(utils::IStream &fi) {
    net_cfg.LoadNet(fi);
    fi.Read(&epoch_counter, sizeof(epoch_counter));
    this->FreeNet();
    this->InitNet();
    fi.Read(&model_blob_);
    for (size_t i = 0; i < nets_.size(); ++i) {
      utils::MemoryBufferStream fs(&model_blob_);
      nets_[i]->LoadModel(fs);
      nets_[i]->WaitJob();
    }
    this->InitTemp();
  }
  virtual void StartRound(int round) {
    for (size_t i = 0; i < nets_.size(); ++i) {
      nets_[i]->StartRound(round);
    }
    this->WaitAllJobs();
  }
  virtual void Update(const DataBatch& data) {
    mshadow::Shape<4> oshape = out_temp.shape_;
    oshape[0] = data.batch_size;
    out_temp.Resize(oshape);

    const size_t ndevice = devices_.size();
    mshadow::index_t step = std::max((batch_size + ndevice - 1) / ndevice, 1UL);

    bool need_sync = sample_counter % update_period == 0;
    bool need_update = (sample_counter + 1) % update_period == 0;

    for (mshadow::index_t i = nets_.size(); i != 0; --i) {
      mshadow::index_t begin = std::min((i - 1) * step, data.batch_size);
      mshadow::index_t end = std::min(i * step, data.batch_size);
      mshadow::Tensor<cpu, 4> mbatch = data.data.Slice(begin, end);
      layer::LabelInfo info;
      info.labels = data.labels + begin;
      info.batch_size = end - begin;
      nets_[i - 1]->TrainForwardBackprop(mbatch, info, out_temp.Slice(begin, end),
                                         false, need_sync, need_update, epoch_counter);
    }
    this->WaitAllJobs();
    // evlauate training loss
    if (eval_train != 0) {
      train_metric.AddEval(out_temp.FlatTo2D(), data.labels);
    }
    if (++sample_counter >= update_period) {
      sample_counter = 0;
      epoch_counter += 1;
    }
  }
  virtual void Predict(std::vector<float> &preds, const DataBatch& data) {
    this->ForwardToTemp(data);
    for (index_t i = 0; i < out_temp.size(0); ++i) {
      preds.push_back(this->TransformPred(out_temp[i][0][0]));
    }
  }
  virtual void PredictRaw(std::vector<std::vector<float> > &preds, const DataBatch& batch) {
    this->ForwardToTemp(batch);
    preds.resize(out_temp.size(0));
    for(index_t i = 0; i < out_temp.size(0); ++i) {
      preds[i].resize(out_temp.size(3));
      for (index_t j = 0; j < out_temp.size(3); ++j) {
        preds[i][j] = out_temp[i][0][0][j];
      }
    }
  }
  virtual void PredictTop(std::vector<std::vector<float> > &preds, const DataBatch& batch) {
    this->ForwardToTemp(batch, -2);
    // TODO (bing): merge with PredictRaw
    preds.resize(out_temp.size(0));
    for(index_t i = 0; i < out_temp.size(0); ++i) {
      preds[i].resize(out_temp.size(3));
      for (index_t j = 0; j < out_temp.size(3); ++j) {
        preds[i][j] = out_temp[i][0][0][j];
      }
    }
  }
  virtual std::string Evaluate(IIterator<DataBatch> *iter_eval, const char* data_name) {
    std::string ret;
    if (eval_train != 0) {
      ret += train_metric.Print("train");
      train_metric.Clear();
    }
    if (iter_eval == NULL) return ret;
    metric.Clear();
    iter_eval->BeforeFirst();
    while (iter_eval->Next()) {
      const DataBatch& batch = iter_eval->Value();
      this->ForwardToTemp(batch);
      metric.AddEval(out_temp.Slice(0, out_temp.size(0) - batch.num_batch_padd).FlatTo2D(), batch.labels);
    }
    ret += metric.Print(data_name);
    return ret;
  }

 private:
  inline float TransformPred(mshadow::Tensor<cpu,1> pred) {
    if (pred.size(0) != 1) {
      return GetMaxIndex(pred);
    } else {
      return pred[0];
    }
  }
  inline static int GetMaxIndex(mshadow::Tensor<cpu,1> pred) {
    index_t maxidx = 0;
    for(index_t i = 1; i < pred.size(0); ++ i) {
      if(pred[i] > pred[maxidx]) maxidx = i;
    }
    return maxidx;
  }
  inline void ForwardToTemp(const DataBatch &data, int layer=-1) {
    const int lid = nets_[0]->net().nodes.size() + layer;
    mshadow::Shape<4> oshape = nets_[0]->net().nodes[lid].data.shape_;
    oshape[0] = data.batch_size;
    out_temp.Resize(oshape);
    const size_t ndevice = devices_.size();
    mshadow::index_t step = std::max((batch_size + ndevice - 1) / ndevice, 1UL);
    for (mshadow::index_t i = nets_.size(); i != 0; --i) {
      mshadow::index_t begin = std::min((i - 1) * step, data.batch_size);
      mshadow::index_t end = std::min(i * step, data.batch_size);
      mshadow::Tensor<cpu, 4> mbatch = data.data.Slice(begin, end);
      nets_[i - 1]->PredictForward(mbatch);
    }
    this->WaitAllJobs();
    // copy results out
    for (mshadow::index_t i = nets_.size(); i != 0; --i) {
      mshadow::index_t begin = std::min((i - 1) * step, data.batch_size);
      mshadow::index_t end = std::min(i * step, data.batch_size);
      nets_[i - 1]->CopyNodeData(layer, out_temp.Slice(begin, end));
    }
    this->WaitAllJobs();
  }

  inline void WaitAllJobs(void) {
    for (size_t i = nets_.size(); i != 0; --i) {
      nets_[i - 1]->WaitJob();
    }
  }
  inline void Save2ModelBlob(void) {
    // save to model blob
    model_blob_.clear();
    utils::MemoryBufferStream fs(&model_blob_);
    nets_[0]->SaveModel(fs);
    nets_[0]->WaitJob();
  }
  inline void InitNet(void) {
    utils::Assert(nets_.size() == 0, "net must be empty before this");
    net_cfg.Configure(cfg);
    if (devices_.size() == 0) devices_.push_back(0);
    size_t ndevice = devices_.size();
    mshadow::index_t step = std::max((batch_size + ndevice - 1) / ndevice, 1UL);
    while (step * (devices_.size() - 1) >= batch_size) {
      devices_.pop_back();
    }
    if (ndevice > devices_.size()) {
      ndevice = devices_.size();
      if (silent == 0) {
        printf("Warning: The number of devices is induce mini-batch=%u\n" \
               "We can equally use %lu devices to cover the batch_size\n", step, ndevice);
      }
    }
    this->InitParamServer();
    for (size_t i = 0; i < ndevice; ++i) {
      nets_.push_back(new NeuralNetThread<xpu>(net_cfg, pserver,
                                               devices_[i], step, i + seed * 100));
    }
    if (silent == 0) {
      printf("finish initialization with %lu devices\n", devices_.size());
    }
  }
  inline void InitParamServer(void) {
    utils::Assert(pserver == NULL, "net must be empty before this");
    if (type_pserver == "UNSPECIFIED") {
      if (devices_.size() <=1) type_pserver = "NONE";
      else type_pserver = "local";
    }
    if (type_pserver != "NONE") {
      pserver = mshadow::ps::CreateSharedModel<xpu, real_t>(type_pserver.c_str());
      for (size_t i = 0; i < cfg.size(); ++i) {
        pserver->SetParam(cfg[i].first.c_str(), cfg[i].second.c_str());
      }
      if (devices_.size() == 0) devices_.push_back(0);
      pserver->Init(devices_);
    }
  }
  inline void InitTemp(void) {
    mshadow::Shape<4> oshape = nets_[0]->net().nodes.back().data.shape_;
    oshape[0] = batch_size;
    out_temp.Resize(oshape);
  }
  inline void FreeNet(void) {
    for (size_t i = 0; i < devices_.size(); ++i) {
      delete nets_[i];
    }
    nets_.clear();
    if (pserver != NULL) {
      delete pserver;
      pserver = NULL;
    }
  }
  /*! \brief parameter server */
  mshadow::ps::ISharedModel<xpu, real_t> *pserver;
  /*! \brief type of parameter server */
  std::string type_pserver;
  /*! \brief epoch counter */
  long epoch_counter;
  /*! \brief seed to the layers */
  int seed;
  /*! \brief silent*/
  int silent;
  /*! \brief update period */
  int update_period;
  /*! \brief sample counter */
  int sample_counter;
  /*! \brief show train eval */
  int eval_train;
  /*! \brief evaluator */
  utils::MetricSet metric;
  /*! \brief evaluator for train */
  utils::MetricSet train_metric;
  /*! \brief final space of output node */
  mshadow::TensorContainer<cpu, 4> out_temp;
  // ------- model part --------
  /*! \brief batch size */
  mshadow::index_t batch_size;
  /*! \brief record the devices_ used in each thread */
  std::vector<int> devices_;
  /*! \brief serialized model in CPU */
  std::string model_blob_;
  /*! \brief threads of neural nets */
  std::vector<NeuralNetThread<xpu>*> nets_;
  /*! \brief network configuration type */
  NetConfig net_cfg;
  /*! \brief history of configurations */
  std::vector< std::pair<std::string, std::string> > cfg;
};

template<typename xpu>
INetTrainer *CreateNet_(int net_type) {
  return new CXXNetThreadTrainer<xpu>();
}
}  // namespace nnet
}  // cxxnet
