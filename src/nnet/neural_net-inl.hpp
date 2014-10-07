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
#include "./nnet_config.h"

namespace cxxnet {
namespace nnet {
/*! \brief implementation of abstract neural net */
template<typename xpu>
struct NeuralNet {
  /*! \brief network configuration configure */
  const NetConfig &cfg;
  /*! \brief label information */
  layer::LabelInfo label_info;
  /*! \brief nodes in the neural net */
  std::vector< Node<xpu> > nodes;
  /*! \brief layers in the neural net */
  std::vector<ILayer<xpu>*> layers;
  /*! \brief updaters in the neural net */
  std::vector<IUpdater<xpu>*> updaters;
  /*! \brief random number generator */
  mshadow::Random<xpu> rnd;

  // constructor do nothing
  NeuralNet(void) : rnd(0) {
  }
  /*! \brief save model to file */
  inline void SaveModel(utils::IStream &fo) const {
    for (int i = 0; i < layers.size(); ++ i) {
      layers[i]->SaveModel(fo);
    }
  }
  /*! \brief initial model parameters in the beginning */
  inline void InitModel(void) {
    this->InitNet();
    this->ConfigLayers();
    for (int i = 0; i < cfg.param.num_layers; ++ i) {
      layers[i]->InitLayer();
    }
    for (int i = 0; i < cfg.param.num_layers; ++ i) {
      layers[i]->InitModel();
    }
    this->InitNodes();
  }
  /*! \brief load model from stream */
  inline void LoadModel(utils::IStream &fi) {
    this->InitNet();
    this->ConfigLayers();
    for (int i = 0; i < cfg.param.num_layers; ++ i) {
      layers[i]->LoadModel(fi);
    }
    for (int i = 0; i < cfg.param.num_layers; ++ i) {
      layers[i]->InitLayer();
    }
    this->InitNodes();
  }
  /*!
   * \brief forward prop
   * \param is_train whether is training phase
   */
  inline void Forward(bool is_train) {
    for (size_t i = 0; i < layers.size(); ++ i) {
      layers[i]->Forward( is_train );
    }
  }

 private:
  inline void InitNet(void) {
    nodes.resize(cfg.param.num_nodes);
    // setup input shape
    Shape<4> ishape;
    ishape[0] = cfg.param.input_shape[0];
    ishape[1] = cfg.param.input_shape[1];
    ishape[2] = cfg.param.input_shape[2];
    ishape[3] = cfg.batch_size;
    nodes[0].shape = ishape;
    // input layer
    for (int i = 0; i < cfg.param.num_layers; ++i) {
      std::vector<Node<xpu>*> nodes_in;
      std::vector<Node<xpu>*> nodes_out;
      const NetConfig::LayerInfo &info =cfg.layers[i];
      for (size_t j = 0; j < info.nindex_in.size() ++j) {
        nodes_in.push_back(&info.nindex_in[j]);
      }
      for (size_t j = 0; j < info.nindex_out.size() ++j) {
        nodes_out.push_back(&info.nindex_out[j]);
      }
      layers.push_back(layer::CreateLayer(&rnd, nodes_in, nodes_out, &label_info));
    }
  }
  inline void ConfigLayers(void) {
    for (int i = 0; i < cfg.param.num_layers; ++ i) {
      for (size_t j = 0; j < cfg.defcfg.size(); ++j) {
        layers[i]->SetParam(cfg.defcfg[j].first.c_ptr(),
                            cfg.defcfg[j].second.c_ptr());      
      }
      for (size_t j = 0; j < cfg.layercfg[i].size(); ++j) {
        layers[i]->SetParam(cfg.layercfg[i][j].first.c_ptr(),
                            cfg.layercfg[i][j].second.c_ptr());
      }
    }    
  }
  inline void InitNodes(void) {
    for (size_t i = 0; i < nodes.size(); ++ i) {
      mshadow::Shape<4> s = nodes[i].data.shape;
      mshadow::AllocSpace(nodes[i].data);
      printf("node[%d].shape: %u,%u,%u,%u\n",(int)i, s[3],s[2],s[1],s[0] );
    }    
  }
  /*! \brief free the space */
  inline void FreeSpace(void) {
    for (size_t i = 0; i < nodes.size(); ++i) {
      nodes[i].FreeSpace();
    }
    for (size_t i = 0; i < layers.size(); ++i) {
      delete layers[i];
    }
    for (size_t i = 0; i < updaters.size(); ++i) {
      delete updaters[i];
    }
    nodes.clear(); layers.clear(); updaters.clear();
  }
};

/*! 
 * \brief neural net that runs with an independent thread
 * backed by NeuralNet
 */
template<typename xpu>
class NeuralNetThread {
  
};

}  // namespace nnet
}  // namespace cxxnet
#endif
