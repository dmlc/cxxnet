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
  }
  virtual ~CXXNetThreadTrainer(void) {
    this->FreeNet();
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "add_device")) {
      devices_.push_back(atoi(val));
    }
    if (!strcmp(name, "batch_size")) batch_size = static_cast<mshadow::index_t>(atoi(val));
    if (!strcmp(name, "update_period")) update_period = atoi(val);
    if (!strcmp( name, "eval_train")) eval_train = atoi(val);
    if( !strcmp( name, "metric") ) {
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
    }
    this->WaitAllJobs();
  }
  virtual void StartRound(int round) {
    for (size_t i = 0; i < nets_.size(); ++i) {
      nets_[i]->StartRound(round);
    }
    this->WaitAllJobs();
  }
  virtual void Update(const DataBatch& data) {
    mshadow::Shape<4> oshape = out_temp.shape; oshape[3] = data.batch_size;
    out_temp.Resize(oshape);

    const size_t ndevice = devices_.size();
    mshadow::index_t step = std::max((batch_size + ndevice - 1) / ndevice, 1UL);
    for (mshadow::index_t i = nets_.size(); i != 0; --i) {
      mshadow::index_t begin = std::min((i - 1) * step, data.batch_size); 
      mshadow::index_t end = std::min(i * step, data.batch_size);
      mshadow::Tensor<cpu, 4> mbatch = data.data.Slice(begin, end);
      layer::LabelInfo info;
      info.labels = data.labels + begin;
      info.batch_size = end - begin;
      nets_[i]->TrainForwardBackprop(mbatch, info, out_temp.Slice(begin, end));
    }
    this->WaitAllJobs();
    // evlauate training loss
    if (eval_train != 0) {
      train_metric.AddEval(out_temp.FlatTo2D(), data.labels);
    }
    if (++ sample_counter >= update_period) {
      for (mshadow::index_t i = nets_.size(); i != 0; --i) {
        nets_[i]->Update(epoch_counter);
      }
      epoch_counter += 1;
      this->WaitAllJobs();
      sample_counter = 0;
    }
  }
  virtual void Predict(std::vector<float> &preds, const DataBatch& data) {
    this->ForwardToTemp(data);
    for (mshadow::index_t i = 0; i < out_temp.shape[3]; ++i) {
      preds.push_back(this->TransformPred(out_temp[3][0][0]));
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
      metric.AddEval(out_temp.Slice(0, out_temp.shape[3]-batch.num_batch_padd).FlatTo2D(), batch.labels);
    }    
    ret += metric.Print(data_name);
    return ret;
  }

 private:
  inline float TransformPred(mshadow::Tensor<cpu,1> pred) {
    if (pred.shape[0] != 1) {
      return GetMaxIndex(pred);
    } else {
      return pred[0];
    }
  }
  inline static int GetMaxIndex( mshadow::Tensor<cpu,1> pred ){
    index_t maxidx = 0;
    for( index_t i = 1; i < pred.shape[0]; ++ i ){
      if( pred[i] > pred[maxidx] ) maxidx = i;
    }
    return maxidx;
  }
  inline void ForwardToTemp(const DataBatch &data) {
    mshadow::Shape<4> oshape = out_temp.shape; oshape[3] = data.batch_size;
    out_temp.Resize(oshape);
    const size_t ndevice = devices_.size();    
    mshadow::index_t step = std::max((batch_size + ndevice - 1) / ndevice, 1UL);
    for (mshadow::index_t i = nets_.size(); i != 0; --i) {
      mshadow::index_t begin = std::min((i - 1) * step, data.batch_size); 
      mshadow::index_t end = std::min(i * step, data.batch_size);
      mshadow::Tensor<cpu, 4> mbatch = data.data.Slice(begin, end);
      nets_[i]->PredictForward(mbatch);
    }    
    this->WaitAllJobs();
    // copy results out
    for (mshadow::index_t i = nets_.size(); i != 0; --i) {
      mshadow::index_t begin = std::min((i - 1) * step, data.batch_size); 
      mshadow::index_t end = std::min(i * step, data.batch_size);
      nets_[i]->CopyNodeData(-1, out_temp.Slice(begin, end));
    }
    this->WaitAllJobs();
  }
  
  inline void WaitAllJobs(void) {
    for (size_t i = nets_.size(); i != 0; --i) {
      nets_[i-1]->WaitJob();
    }
  }
  inline void Save2ModelBlob(void){
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
    const size_t ndevice = devices_.size();
    mshadow::index_t step = std::max((batch_size + ndevice - 1) / ndevice, 1UL);
    for (size_t i = 0; i < ndevice; ++i) {
      nets_.push_back(new NeuralNetThread<xpu>(net_cfg, devices_[i], step));
    }
    if (silent == 0) {
      printf("finish initialization with %lu devices\n", devices_.size());
    }
    mshadow::Shape<4> oshape = nets_[0]->net().nodes.back().data.shape;    
    oshape[3] = batch_size;
    out_temp.Resize(oshape);
  }
  inline void FreeNet(void) {
    for (size_t i = 0; i < devices_.size(); ++i) {
      delete nets_[i];
    }
    nets_.clear();
  }
  
  /*! \brief epoch counter */
  long epoch_counter;
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
  /*! \brief threads of */
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
