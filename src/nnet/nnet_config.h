#ifndef CXXNET_NNET_NNET_CONFIG_H_
#define CXXNET_NNET_NNET_CONFIG_H_
/*!
 * \file cxxnet_net-inl.hpp
 * \brief implementation of netural nets
 * \author Tianqi Chen, Bing Xu
 */
#include <vector>
#include <utility>
#include <string>
#include <cstring>
#include <mshadow/tensor.h>
#include "../layer/layer.h"
#include "../utils/utils.h"
#include "../utils/io.h"

namespace cxxnet {
namespace nnet {
/*! 
 * \brief this is an object that records the configuration of a neural net
 *    it is used to store the network structure, and reads in configuration 
 *    that associates with each of the layers
 */
struct NetConfig {
  /*! \brief general model parameter */
  struct NetParam {
    /*! \brief number of nodes in the network */
    int num_nodes;
    /*! \brief number of layers in the network */
    int num_layers;
    /*! \brief input shape, not including batch dimension */
    mshadow::Shape<3> input_shape;
    /*! \brief whether the configuration is finalized and the network structure is fixed */
    int init_end;
    /*! \brief reserved fields, used to extend data structure */
    int reserved[32];
    /*! \brief constructor */
    NetParam(void) {
      memset(reserved, 0, sizeof(reserved));
      num_nodes = 0;
      num_layers = 0;      
      input_shape = mshadow::Shape3(0, 0, 0);
      init_end = 0;
    }
  };
  /*! \brief information about each layer */
  struct LayerInfo {
    /*! \brief type of layer */
    layer::LayerType type;
    /*! \brief input node index */
    std::vector<int> nindex_in;
    /*! \brief output node node index */    
    std::vector<int> nindex_out;
    /*! \brief equality check */
    inline bool operator==(const LayerInfo &b) const {
      if (type != b.type ||
          nindex_in.size() != b.nindex_in.size() || 
          nindex_out.size() != b.nindex_out.size())  return false;
      for (size_t i = 0; i < nindex_in.size(); ++i) {
        if (nindex_in[i] != b.nindex_in[i]) return false;
      }
      for (size_t i = 0; i < nindex_out.size(); ++i) {
        if (nindex_out[i] != b.nindex_out[i]) return false;
      }
      return true;
    }
  };
  // model parameters that defines network configuration
  /*! \brief generic parameters about net */
  NetParam param;
  /*! \brief per layer information */
  std::vector<LayerInfo> layers;
  // -----------------------------
  // Training parameters that can be changed each time, even when network is fixed
  // the training parameters will not be saved during LoadNet SaveNet
  //
  /*! \brief batch size used for training */
  int batch_size;
  /*! \brief type of updater function */
  std::string updater_type;
  /*! \brief default global configuration */
  std::vector< std::pair< std::string, std::string > > defcfg;
  /*! \brief extra parameter configuration specific to this layer */
  std::vector< std::vector< std::pair<std::string, std::string> > > layercfg;
  // constructor
  NetConfig(void) {
    // default training parameters
    batch_size = 100;
    updater_type = "sgd";
    netcfg_mode = 0;
    cfg_top_node = 0;
    cfg_layer_index = 0;
  }
  /*!
   * \brief save network structure to output
   *  note: this operation does not save the training configurations
   *        such as updater_type, batch_size
   * \param fo output stream
   */
  inline void SaveNet(utils::IStream &fo) const {
    fo.Write(&param, sizeof(param));
    utils::Assert(param.num_layers == static_cast<int>(layers.size()),
                  "model inconsistent");
    for (int i = 0; i < param.num_layers; ++i) {
      fo.Write(&layers[i].type, sizeof(layer::LayerType));
      fo.Write(layers[i].nindex_in);
      fo.Write(layers[i].nindex_out);
    } 
  }
  /*!
   * \brief save network structure from input
   *  note: this operation does not load the training configurations
   *        such as updater_type, batch_size
   * \param fi output stream
   */
  inline void LoadNet(utils::IStream &fi) {
    utils::Check(fi.Read(&param, sizeof(param)) != 0,
                 "NetConfig: invalid model file");
    layers.resize(param.num_layers);
    layercfg.resize(param.num_layers);
    for (int i = 0; i < param.num_layers; ++i) {
      utils::Check(fi.Read(&layers[i].type, sizeof(layer::LayerType)) != 0,
                 "NetConfig: invalid model file");                   
      utils::Check(fi.Read(&layers[i].nindex_in), "NetConfig: invalid model file");
      utils::Check(fi.Read(&layers[i].nindex_out), "NetConfig: invalid model file");
    }    
  }
  /*! \brief guess parameters, from current setting, this will set init_end in param to be true */
  inline void InitNet(void) {
    param.num_nodes = 0;
    param.num_layers = static_cast<int>(layers.size());
    for (size_t i = 0; i < layers.size(); ++ i) {
      const LayerInfo &info = layers[i];
      for (size_t j = 0; j < info.nindex_in.size(); ++j) {
        param.num_nodes = std::max(info.nindex_in[j] + 1, param.num_nodes);
      }
      for (size_t j = 0; j < info.nindex_out.size(); ++j) {
        param.num_nodes = std::max(info.nindex_out[j] + 1, param.num_nodes);
      }
    }
    param.init_end = 1;
  }
  /*!
   * \brief Set param for the layer from string
   * \param name parameter name
   * \param val string for configuration
   */
  inline void SetParam(const char *name, const char *val) {
    if (param.init_end == 0) { 
      if (!strcmp( name, "input_shape")) {
        unsigned x, y, z;
        utils::Check(sscanf(val, "%u,%u,%u", &z, &y, &x) == 3,
                     "input_shape must be three consecutive integers without space example: 1,1,200 ");
        param.input_shape = mshadow::Shape3(z, y, x);
      }
    }
    if (!strcmp(name, "batch_size")) batch_size = atoi(val);
    if (!strcmp(name, "updater")) updater_type = val;
    if (!strcmp(name, "netconfig") && !strcmp(val, "start")) netcfg_mode = 1;
    if (!strcmp(name, "netconfig") && !strcmp(val, "end"))   netcfg_mode = 0;    
    if (!strncmp(name, "layer[", 6)) {
      LayerInfo info = this->GetLayerInfo(name, val, cfg_top_node);
      netcfg_mode = 2;
      if (param.init_end == 0) {
        utils::Assert(layers.size() == static_cast<size_t>(cfg_layer_index), "NetConfig inconsistent");
        layers.push_back(info);
        layercfg.resize(layers.size());
        param.num_layers = static_cast<int>(layers.size());
      } else {
        utils::Check(cfg_layer_index < static_cast<int>(layers.size()) &&
                     info == layers[cfg_layer_index],
                     "config setting does not match existing network structure");
      }
      if (info.nindex_out.size() != 0) {
        cfg_top_node = info.nindex_out[0];
      }
      cfg_layer_index += 1;
    }
    if (netcfg_mode == 2) {
      layercfg[cfg_layer_index - 1].push_back(std::make_pair(std::string(name), std::string(val)));
    } else {
      defcfg.push_back(std::make_pair(std::string(name), std::string(val)));
    }
  }
  
 private:
  // configuration parser to parse layer info, support one to to one connection for now
  // extend this later to support multiple connections
  inline LayerInfo GetLayerInfo(const char *name, const char *val, int top_node) {
    LayerInfo inf;
    int a, b;
    char ltype[256], tag[256];
    if (sscanf(name, "layer[%d->%d]", &a, &b) != 2) {
      inf.nindex_in.push_back(a);
      inf.nindex_out.push_back(b);
    } else if (sscanf(name, "layer[+%d]", &b) == 1) {
      a = top_node; b += top_node;
      inf.nindex_in.push_back(a);
      inf.nindex_out.push_back(b);      
    } else {
      utils::Error("invalid config format");
    }
    utils::Check(sscanf(val , "%[^:]:%s", ltype, tag) >= 1, "invalid config format");
    inf.type = layer::GetLayerType(ltype);    
    return inf;
  }
  // auxiliary data structure to help read configuration
  // whether in net config mode
  int netcfg_mode;
  // remembers what is the last top node
  int cfg_top_node;
  // current configuration layer index
  int cfg_layer_index;
};
}  // namespace nnet
}  // namespace cxxnet
#endif
