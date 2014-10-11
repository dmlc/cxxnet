#ifndef CXXNET_NNET_NEURAL_NET_INL_HPP_
#define CXXNET_NNET_NEURAL_NET_INL_HPP_
/*!
 * \file neural_net-inl.hpp
 * \brief implementation of common neuralnet
 * \author Tianqi Chen
 */
#include <vector>
#include <utility>
#include <mshadow/tensor.h>
#include "../layer/layer.h"
#include "../updater/updater.h"
#include "../utils/utils.h"
#include "../utils/io.h"
#include "../utils/thread.h"
#include "./nnet_config.h"

namespace cxxnet {
namespace nnet {
/*! \brief implementation of abstract neural net */
template<typename xpu>
struct NeuralNet {
  /*! \brief network configuration configure */
  const NetConfig &cfg;
  /*! \brief maximum batch_size */
  mshadow::index_t max_batch;
  /*! \brief label information */
  layer::LabelInfo label_info;
  /*! \brief nodes in the neural net */
  std::vector<layer::Node<xpu> > nodes;
  /*! \brief layers in the neural net */
  std::vector<layer::Connection<xpu> > connections;
  /*! \brief updaters in the neural net */
  std::vector< std::vector<updater::IUpdater<xpu>*> > updaters;
  /*! \brief random number generator */
  mshadow::Random<xpu> rnd;
  // constructor do nothing
  NeuralNet(const NetConfig &cfg, mshadow::index_t batch_size)       
      : cfg(cfg), rnd(0) {
    // set maximum batch
    this->max_batch = batch_size;
  }
  ~NeuralNet(void) {
    this->FreeSpace();
  }
  /*! \brief save model to file */
  inline void SaveModel(utils::IStream &fo) const {
    for (int i = 0; i < connections.size(); ++ i) {
      if (connections[i].type != layer::kSharedLayer) {
        connections[i].layer->SaveModel(fo);
      }
    }
  }
  /*! \brief initial model parameters in the beginning */
  inline void InitModel(void) {
    this->InitNet();
    this->ConfigConntions();
    for (size_t i = 0; i < connections.size(); ++ i) {
      layer::Connection<xpu> &c = connections[i];
      c.layer->InitConnection(c.nodes_in, c.nodes_out, &c.state);
    }
    for (size_t i = 0; i < connections.size(); ++ i) {
      if (connections[i].type != layer::kSharedLayer) {        
        connections[i].layer->InitModel();
      }
    }
    this->InitNodes();
    this->InitUpdaters();
  }
  /*! \brief load model from stream */
  inline void LoadModel(utils::IStream &fi) {
    this->InitNet();
    this->ConfigConntions();
    for (size_t i = 0; i < connections.size(); ++ i) {
      if (connections[i].type != layer::kSharedLayer) {
        connections[i].layer->LoadModel(fi);
      }
    }    
    for (size_t i = 0; i < connections.size(); ++ i) {
      layer::Connection<xpu> &c = connections[i];
      c.layer->InitConnection(c.nodes_in, c.nodes_out, &c.state);
    }
    this->InitNodes();
    this->InitUpdaters();
  }
  /*!
   * \brief forward prop
   * \param is_train whether is training phase
   * \param batch the input batch
   */
  inline void Forward(bool is_train, mshadow::Tensor<cpu,4> batch) {
    // check if we need to adjust batch size according to the input
    this->AdjustBatchSize(batch.shape[3]);
    // copy data into node
    mshadow::Copy(nodes[0].data, batch);
    for (size_t i = 0; i < connections.size(); ++i) {
      layer::Connection<xpu> &c = connections[i];
      c.layer->Forward(is_train, c.nodes_in, c.nodes_out, &c.state);
    }
  }
  /*! 
   * \brief backprop 
   * \param prop_to_input whether prop gradient to input node
   */
  inline void Backprop(bool prop_to_input = false) {    
    for (size_t i = connections.size(); i > 0; --i) {
      layer::Connection<xpu> &c = connections[i-1];
      c.layer->Backprop(i != 1 || prop_to_input,
                        c.nodes_in, c.nodes_out, &c.state);
    }
  }
  /*!
   * \brief update model parameters 
   * \param epoch number of epoches
   */
  inline void Update(size_t epoch) {
    for (size_t i = 0; i < updaters.size(); ++ i) {
      for (size_t j = 0; j < updaters[i].size(); ++ j) {
        updaters[i][j]->Update(epoch);
      }
    }
  }
  /*!
   * \brief notify round start
   * \param round round counter
   */
  inline void StartRound(int round) {
    for (size_t i = 0; i < updaters.size(); ++ i) {
      for (size_t j = 0; j < updaters[i].size(); ++ j) {
        updaters[i][j]->StartRound(round);
      }
    }
  }
  
 private:
  // intialize the neural net data structure
  inline void InitNet(void) {
    nodes.resize(cfg.param.num_nodes);
    mshadow::Shape<3> s = cfg.param.input_shape;
    // setup input shape
    nodes[0].data.shape = mshadow::Shape4(max_batch, s[2], s[1], s[0]);
    // input layer
    for (int i = 0; i < cfg.param.num_layers; ++i) {
      const NetConfig::LayerInfo &info = cfg.layers[i];
      layer::Connection<xpu> c;
      c.type = info.type;
      for (size_t j = 0; j < info.nindex_in.size(); ++j) {
        c.nodes_in.push_back(&nodes[info.nindex_in[j]]);
      }
      for (size_t j = 0; j < info.nindex_out.size(); ++j) {
        c.nodes_out.push_back(&nodes[info.nindex_out[j]]);
      }
      if (c.type == layer::kSharedLayer) {
        utils::Assert(info.primary_layer_index >=0, "primary_layer_index problem");
        utils::Check(info.primary_layer_index < static_cast<int>(connections.size()), 
                     "shared layer primary_layer_index exceed bound");
        c.layer = connections[info.primary_layer_index].layer;
      } else {
        c.layer = layer::CreateLayer(c.type, &rnd, &label_info);
      }
      connections.push_back(c);
    }
  }
  // configure the parameters of layer
  inline void ConfigConntions(void) {
    for (int i = 0; i < cfg.param.num_layers; ++ i) {
      if (connections[i].type == layer::kSharedLayer) continue;
      for (size_t j = 0; j < cfg.defcfg.size(); ++j) {
        connections[i].layer->SetParam(cfg.defcfg[j].first.c_str(),
                                       cfg.defcfg[j].second.c_str());      
      }
      for (size_t j = 0; j < cfg.layercfg[i].size(); ++j) {
        connections[i].layer->SetParam(cfg.layercfg[i][j].first.c_str(),
                                       cfg.layercfg[i][j].second.c_str());
      }
    }
  }
  // create the updaters
  inline void InitUpdaters(void) {
    for (int i = 0; i < cfg.param.num_layers; ++ i) {
      if (connections[i].type == layer::kSharedLayer) continue;
      std::vector<updater::IUpdater<xpu>*> out;
      updater::CreateUpdaters(cfg.updater_type.c_str(),
                              &rnd, connections[i].layer, &out);
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
      }
      updaters.push_back(out);
    }
  }
  // intialize the space of nodes
  inline void InitNodes(void) {
    for (size_t i = 0; i < nodes.size(); ++ i) {
      mshadow::Shape<4> s = nodes[i].data.shape;
      mshadow::AllocSpace(nodes[i].data);
      printf("node[%lu].shape: %u,%u,%u,%u\n", i, s[3], s[2], s[1], s[0]);
    }
  }
  // adjust batch size to a new value, the batch_size must be smaller than max_batch
  inline void AdjustBatchSize(mshadow::index_t batch_size) {
    utils::Assert(max_batch >= batch_size, "cannot set batch size larger than max batch");
    if (batch_size != nodes[0].data.shape[3]) {
      for (size_t i = 0; i < nodes.size(); ++i) {
        nodes[i].data.shape[3] = batch_size;
      }
      for (size_t i = 0; i < connections.size(); ++ i) {
        layer::Connection<xpu> &c = connections[i];
        c.layer->OnBatchSizeChanged(c.nodes_in, c.nodes_out, &c.state);  
      }
    }
  }
  /*! \brief free all space allocated in this struct*/
  inline void FreeSpace(void) {
    for (size_t i = 0; i < nodes.size(); ++i) {
      nodes[i].FreeSpace();
    }
    for (size_t i = 0; i < connections.size(); ++i) {
      if (connections[i].type != layer::kSharedLayer) {        
        delete connections[i].layer;
      }
    }
    for (size_t i = 0; i < updaters.size(); ++i) {
      for (size_t j = 0; j < updaters[i].size(); ++j) {
        delete updaters[i][j];
      }
    }
    nodes.clear(); connections.clear(); updaters.clear();
  }
};

/*!
 * \brief neural net that runs with an independent thread backed by NeuralNet
 * \tparam
 */
template<typename xpu>
class NeuralNetThread {
 public:
  /*! \brief create a new neural net thread on specific device */
  NeuralNetThread(const NetConfig &cfg, 
                  int device_id,
                  mshadow::index_t batch_size,
                  bool new_thread = true)
      : cfg(cfg), device_id(device_id),
        batch_size(batch_size), new_thread(new_thread) {
    net_ = NULL;
    if (new_thread) {
      destroy_signal = false;
      job_start.Init(0);
      job_end.Init(0);
      worker_thread.Start(ThreadEntry, this);
      // wait until net is created
      job_end.Wait();
    } else {
      if (!xpu::kDevCPU) {
        mshadow::InitTensorEngine(device_id);
      }
      net_ = new NeuralNet<xpu>(cfg, batch_size);
    }
  }
  // destructor
  ~NeuralNetThread(void) {
    if (net_ != NULL) {
      if (new_thread) {
        destroy_signal = true;
        job_start.Post();
        worker_thread.Join();
        job_start.Destroy();
        job_end.Destroy();
      } else {
        delete net_;
        if (!xpu::kDevCPU) {
          mshadow::ShutdownTensorEngine();
        }
      }
    }
  }

  /*! 
   * \brief wait till the the thread finishes current task 
   * This function MUST be called every time before running next job
   */
  inline void WaitJob(void) {
    if (new_thread) job_end.Wait();
  }
  inline void InitModel(void) {
    this->task = kInitModel;
    this->ExecTask();
  }
  inline void SaveModel(utils::IStream &fo) {
    iparam_fp = &fo;
    this->task = kSaveModel;
    this->ExecTask();
  }
  inline void LoadModel(utils::IStream &fi) {
    iparam_fp = &fi;
    this->task = kLoadModel;
    this->ExecTask();
  }
  inline void Update(size_t epoch) {
    iparam_epoch = epoch;
    this->task = kUpdate;
    this->ExecTask();
  }
  inline void StartRound(int round) {
    iparam_epoch = static_cast<size_t>(round);
    this->task = kStartRound;
    this->ExecTask();
  }
  /*! \brief run a training forward backprop pass */
  inline void TrainForwardBackprop(mshadow::Tensor<cpu,4> batch,
                                   const layer::LabelInfo &label_info,
                                   mshadow::Tensor<cpu,4> out_data,
                                   bool prop_to_input = false) {
    utils::Assert(net_ != NULL, "thread must be initialized before use");
    net_->label_info = label_info;
    iparam_batch = batch;
    iparam_flag = prop_to_input;
    oparam_node = out_data;
    this->task = kTrainProp;
    this->ExecTask();
  }
  /*! \brief run a predicting forward pass, copy final layer  */
  inline void PredictForward(mshadow::Tensor<cpu,4> batch) {
    iparam_batch = batch;
    this->task = kPredForward;
    this->ExecTask();
  }
  // copy node data out
  inline void CopyNodeData(int nid, mshadow::Tensor<cpu,4> out_data) {
    iparam_nid = nid;
    oparam_node = out_data;
    this->task = kCopyNode;
    this->ExecTask();
  }
  // return reference of node
  inline const NeuralNet<xpu> &net(void) const{
    return *net_;
  }

 private:
  // type of task that can be executed
  enum TaskType {
    kInitModel,
    kLoadModel,
    kSaveModel,
    kUpdate,
    kStartRound,
    kTrainProp,
    kPredForward,
    kCopyNode
  };
  // thread related code
  inline static CXXNET_THREAD_PREFIX ThreadEntry(void *pthread) {
    static_cast<NeuralNetThread<xpu>*>(pthread)->RunThread();
    utils::ThreadExit(NULL);
    return NULL;
  }
  inline void RunThread(void) {
    if (!xpu::kDevCPU) {
      mshadow::InitTensorEngine(device_id);
    }
    // allocate net
    net_ = new NeuralNet<xpu>(cfg, batch_size);
    // tell the master that net is created
    job_end.Post();
    while (!destroy_signal) {
      job_start.Wait();
      if (destroy_signal) break;
      this->TaskDispatch();
      job_end.Post();
    }
    delete net_;
    if (!xpu::kDevCPU) {
      mshadow::ShutdownTensorEngine();
    }
  }
  inline void ExecTask(void) {
    if (new_thread) {
      job_start.Post();
    } else {
      this->TaskDispatch();
    }
  }
  inline void TaskDispatch(void) {
    utils::Assert(net_ != NULL, "thread must be initialized before use");
    switch (task) {
      case kInitModel: net_->InitModel(); return;
      case kLoadModel: net_->LoadModel(*iparam_fp); return;
      case kSaveModel: net_->SaveModel(*iparam_fp); return;
      case kUpdate: net_->Update(iparam_epoch); return;
      case kStartRound: net_->StartRound(static_cast<int>(iparam_epoch)); return;
      case kTrainProp: {
        if (iparam_batch.shape[3] == 0) return;
        net_->Forward(true, iparam_batch);
        if (oparam_node.dptr != NULL) {
          mshadow::Copy(oparam_node, net_->nodes.back().data);
        }
        net_->Backprop(iparam_flag);
        return;
      }
      case kPredForward: {
        net_->Forward(false, iparam_batch);
        return;
      }
      case kCopyNode: {
        if (iparam_nid < 0) iparam_nid += static_cast<int>(net_->nodes.size());
        utils::Assert(iparam_nid < static_cast<int>(net_->nodes.size()), "nid out of range");
        mshadow::Copy(oparam_node, net_->nodes[iparam_nid].data);
        return;
      }
    }
  }
  // the following are fields that are used to pass parameters in or out
  // used to copy out fields in the last layer
  mshadow::Tensor<cpu,4> oparam_node;
  // input flag
  bool iparam_flag;  
  // input epochs
  size_t iparam_epoch;
  // input node id
  int iparam_nid;
  // input parameters of file pointers
  utils::IStream *iparam_fp;
  // input batch
  mshadow::Tensor<cpu,4> iparam_batch;
  // current task
  TaskType task;
  // intenal net implementation
  NeuralNet<xpu> *net_;
  // configuration
  const NetConfig &cfg;
  // signal the destruction of object
  bool destroy_signal;
  // signal of jobs
  utils::Semaphore job_end, job_start;
  // thread object
  utils::Thread worker_thread;
  // device id used to intialize tensor engine
  int device_id;
  // local batch size of this thread
  mshadow::index_t batch_size;
  // whether the implementation is backed by a new thread
  const bool new_thread;
};
}  // namespace nnet
}  // namespace cxxnet
#endif
