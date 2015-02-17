#ifndef CXXNET_NNET_NNET_CONFIG_H_
#define CXXNET_NNET_NNET_CONFIG_H_
/*!
 * \file nnet_config.h
 * \brief network structure configuration
 * \author Tianqi Chen, Bing Xu
 */
#include <vector>
#include <utility>
#include <string>
#include <cstring>
#include <map>
#include <mshadow/tensor.h>
#include "../layer/layer.h"
#include "../utils/utils.h"
#include "../utils/io.h"
// #include "glog/logging.h"

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
    /*! \brief the number of extra data */
    int extra_data_num;
    /*! \brief reserved fields, used to extend data structure */
    int reserved[31];
    /*! \brief constructor */
    NetParam(void) {
      memset(reserved, 0, sizeof(reserved));
      num_nodes = 0;
      num_layers = 0;
      input_shape = mshadow::Shape3(0, 0, 0);
      init_end = 0;
      extra_data_num = 0;
    }
  };
  /*! \brief information about each layer */
  struct LayerInfo {
    /*! \brief type of layer */
    layer::LayerType type;
    /*!
     * \brief the index of primary layer,
     *  this field is only used when layer type is kSharedLayer
     */
    int primary_layer_index;
    /*! \brief layer name */
    std::string name;
    /*! \brief input node index */
    std::vector<int> nindex_in;
    /*! \brief output node node index */
    std::vector<int> nindex_out;
    LayerInfo(void) : primary_layer_index(-1), name() {
    }
    /*! \brief equality check */
    inline bool operator==(const LayerInfo &b) const {
      if (type != b.type ||
          primary_layer_index != b.primary_layer_index ||
          nindex_in.size() != b.nindex_in.size() ||
          nindex_out.size() != b.nindex_out.size())  return false;
      if (name != b.name) return false;
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
  /*! \brief name of each node */
  std::vector<std::string> node_names;
  // -----------------------------
  // Training parameters that can be changed each time, even when network is fixed
  // the training parameters will not be saved during LoadNet SaveNet
  //
  /*! \brief maps node name to node index */
  std::map<std::string, int> node_name_map;
  /*! \brief maps tag to layer index */
  std::map<std::string, int> layer_name_map;
  /*! \brief type of updater function */
  std::string updater_type;
  /*! \brief type of synchronization function */
  std::string sync_type;
  /*! \brief map to map input sequence to label name*/
  std::map<std::string, size_t> label_name_map;
  /*! \brief vector to record range of input label*/
  std::vector<std::pair<index_t, index_t> > label_range;
  /*! \brief default global configuration */
  std::vector< std::pair< std::string, std::string > > defcfg;
  /*! \brief extra parameter configuration specific to this layer */
  std::vector< std::vector< std::pair<std::string, std::string> > > layercfg;
  /*! \brief stores the shape of extra data */
  std::vector<int> extra_shape;
  // constructor
  NetConfig(void) {
    updater_type = "sgd";
    sync_type = "simple";
    label_name_map["label"] = 0;
    label_range.push_back(std::make_pair(0, 1));
  }
  /*!
   * \brief save network structure to output
   *  note: this operation does not save the training configurations
   *        such as updater_type, batch_size
   * \param fo output stream
   */
  inline void SaveNet(utils::IStream &fo) const {
    fo.Write(&param, sizeof(param));
    if (param.extra_data_num != 0) {
      fo.Write(extra_shape);
    }
    utils::Assert(param.num_layers == static_cast<int>(layers.size()),
                  "model inconsistent");
    utils::Assert(param.num_nodes == static_cast<int>(node_names.size()),
                  "num_nodes is inconsistent with node_names");
    for (int i = 0; i < param.num_nodes; ++i) {
      fo.Write(node_names[i]);
    }
    for (int i = 0; i < param.num_layers; ++i) {
      fo.Write(&layers[i].type, sizeof(layer::LayerType));
      fo.Write(&layers[i].primary_layer_index, sizeof(int));
      fo.Write(layers[i].name);
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
    node_names.resize(param.num_nodes);
    if (param.extra_data_num != 0) {
      utils::Check(fi.Read(&extra_shape) != 0,
        "NetConfig: Reading extra data shape failed.");
    }
    for (int i = 0; i < param.num_nodes; ++i) {    
      utils::Check(fi.Read(&node_names[i]),
                   "NetConfig: invalid model file");
    }
    node_name_map.clear();
    for (size_t i = 0; i < node_names.size(); ++i) {
      node_name_map[node_names[i]] = static_cast<int>(i);
    }
    layers.resize(param.num_layers);
    layercfg.resize(param.num_layers);
    layer_name_map.clear();
    for (int i = 0; i < param.num_layers; ++i) {
      utils::Check(fi.Read(&layers[i].type, sizeof(layer::LayerType)) != 0,
                 "NetConfig: invalid model file");
      utils::Check(fi.Read(&layers[i].primary_layer_index, sizeof(int)) != 0,
                 "NetConfig: invalid model file");
      utils::Check(fi.Read(&layers[i].name), "NetConfig: invalid model file");
      utils::Check(fi.Read(&layers[i].nindex_in), "NetConfig: invalid model file");
      utils::Check(fi.Read(&layers[i].nindex_out), "NetConfig: invalid model file");
      if (layers[i].type == layer::kSharedLayer) {
        utils::Check(layers[i].name.length() == 0, "SharedLayer must not have name");
      } else {
        if (layers[i].name != "") {
          utils::Check(layer_name_map.count(layers[i].name) == 0,
                       "NetConfig: invalid model file, duplicated layer name: %s",
                       layers[i].name.c_str());
          layer_name_map[layers[i].name] = i;
        }
      }
    }
    this->ClearConfig();
  }
  inline void SetGlobalParam(const char *name, const char *val) {
    if (!strcmp(name, "updater")) updater_type = val;
    if (!strcmp(name, "sync")) sync_type = val;
    {
      unsigned a, b;
      if (sscanf(name, "label_vec[%u,%u)", &a, &b) == 2) {
        label_range.push_back(std::make_pair((index_t)a,
                                             (index_t)b));
        label_name_map[val] = label_range.size() - 1;
      }
    }
  }
  /*!
   * \brief setup configuration, using the config string pass in
   */
  inline void Configure(const std::vector< std::pair<std::string, std::string> > &cfg) {
    this->ClearConfig();
    if (node_names.size() == 0 && node_name_map.size() == 0) {
      node_names.push_back(std::string("in"));
      node_name_map["in"] = 0;
    }
    node_name_map["0"] = 0;
    // whether in net config mode
    int netcfg_mode = 0;
    // remembers what is the last top node
    int cfg_top_node = 0;
    // current configuration layer index
    int cfg_layer_index = 0;
    for (size_t i = 0; i < cfg.size(); ++i) {
      const char *name = cfg[i].first.c_str();
      const char *val = cfg[i].second.c_str();
      if (!strcmp(name, "extra_data_num")) {
        int num;
        sscanf(val, "%d", &num);
        for (int i = 0; i < num; ++i) {
          char name[256];
          sprintf(name, "in_%d", i + 1);
          if (node_name_map.find(name) == node_name_map.end()) {
            node_names.push_back(name);
            node_name_map[name] = i + 1;
          }
        }
        param.extra_data_num = num;
      }
      if (!strncmp(name, "extra_data_shape[", 17)) {
        int extra_num;
        int x, y, z;
        utils::Check(sscanf(name, "extra_data_shape[%d", &extra_num) == 1,
          "extra data shape config incorrect");
        utils::Check(sscanf(val, "%d,%d,%d", &x, &y, &z) == 3,
          "extra data shape config incorrect");
        extra_shape.push_back(x);
        extra_shape.push_back(y);
        extra_shape.push_back(z);
      }
      if (param.init_end == 0) {
        if (!strcmp( name, "input_shape")) {
          unsigned x, y, z;
          utils::Check(sscanf(val, "%u,%u,%u", &z, &y, &x) == 3,
                       "input_shape must be three consecutive integers without space example: 1,1,200 ");
          param.input_shape = mshadow::Shape3(z, y, x);
        }
      }
      if (netcfg_mode != 2) this->SetGlobalParam(name, val);
      if (!strcmp(name, "netconfig") && !strcmp(val, "start")) netcfg_mode = 1;
      if (!strcmp(name, "netconfig") && !strcmp(val, "end")) netcfg_mode = 0;
      if (!strncmp(name, "layer[", 6)) {
        LayerInfo info = this->GetLayerInfo(name, val, cfg_top_node, cfg_layer_index);
        netcfg_mode = 2;
        if (param.init_end == 0) {
          utils::Assert(layers.size() == static_cast<size_t>(cfg_layer_index),
                        "NetConfig inconsistent");
          layers.push_back(info);
          layercfg.resize(layers.size());
        } else {
          utils::Check(cfg_layer_index < static_cast<int>(layers.size()),
                       "config layer index exceed bound");
          utils::Check(info == layers[cfg_layer_index],
                       "config setting does not match existing network structure");
        }
        if (info.nindex_out.size() == 1) {
          cfg_top_node = info.nindex_out[0];
        } else {
          cfg_top_node = -1;
        }
        cfg_layer_index += 1;
        continue;
      }
      if (netcfg_mode == 2) {
        utils::Check(layers[cfg_layer_index - 1].type != layer::kSharedLayer,
                     "please do not set parameters in shared layer, set them in primary layer");
        layercfg[cfg_layer_index - 1].push_back(std::make_pair(std::string(name), std::string(val)));
      } else {
        defcfg.push_back(std::make_pair(std::string(name), std::string(val)));
      }
    }
    if (param.init_end == 0) this->InitNet();
  }

 private:
  // configuration parser to parse layer info, support one to to one connection for now
  // extend this later to support multiple connections
  inline LayerInfo GetLayerInfo(const char *name, const char *val,
                                int top_node, int cfg_layer_index) {
    LayerInfo inf;
    int inc;
    char ltype[256], tag[256];
    char src[256], dst[256];
    if (sscanf(name, "layer[+%d", &inc) == 1) {
      utils::Check(top_node >=0,
                   "ConfigError: layer[+1] is used, "\
                   "but last layer have more than one output"\
                   "use layer[input-name->output-name] instead");
      inf.nindex_in.push_back(top_node);
      if (sscanf(name, "layer[+1:%[^]]]", tag) == 1) {
        inf.nindex_out.push_back(GetNodeIndex(tag, true));
      } else {
        if (inc == 0) {
          inf.nindex_out.push_back(top_node);
        } else {
          utils::SPrintf(tag, sizeof(tag), "!node-after-%d", top_node);
          inf.nindex_out.push_back(GetNodeIndex(tag, true));
        }
      }
    } else if (sscanf(name, "layer[%[^-]->%[^]]]", src, dst) == 2) {
      this->ParseNodeIndex(src, &inf.nindex_in, false);
      this->ParseNodeIndex(dst, &inf.nindex_out, true);
    } else {
      utils::Error("ConfigError: invalid layer format %s", name);
    }
    std::string s_tag, layer_name;
    if (sscanf(val , "%[^:]:%s", ltype, tag) == 2) {
      inf.type = layer::GetLayerType(ltype);
      layer_name = tag;
    } else {
      inf.type = layer::GetLayerType(val);
    }
    if (inf.type == layer::kSharedLayer) {
      const char* layer_type_start = strchr(ltype, '[');
      utils::Check(layer_type_start != NULL,
                   "ConfigError: shared layer must specify tag of layer to share with");
      s_tag = layer_type_start + 1;
      s_tag = s_tag.substr(0, s_tag.length() - 1);
      utils::Check(layer_name_map.count(s_tag) != 0,
                   "ConfigError: shared layer tag %s is not defined before", s_tag.c_str());
      inf.primary_layer_index = layer_name_map[s_tag];
    } else {
      if (layer_name.length() != 0) {
        if (layer_name_map.count(layer_name) != 0) {
          utils::Check(layer_name_map[layer_name] == cfg_layer_index,
                       "ConfigError: layer name in the configuration file do not "\
                       "match the name stored in model");
        } else {
          layer_name_map[layer_name] = cfg_layer_index;
        }
        inf.name = layer_name;
      }
    }
    return inf;
  }
  inline void ParseNodeIndex(char *nodes,
                             std::vector<int> *p_indexs,
                             bool alloc_unknown) {
    char *pch = strtok(nodes, ",");
    while (pch != NULL) {
      p_indexs->push_back(GetNodeIndex(pch, alloc_unknown));
      pch = strtok(NULL, ",");
    }
  }
  inline int GetNodeIndex(const char *name, bool alloc_unknown) {
    std::string key = name;
    std::map<std::string, int>::iterator it
        = node_name_map.find(key);
    if (it == node_name_map.end() || key != it->first) {
      utils::Check(alloc_unknown,
                   "ConfigError: undefined node name %s,"\
                   "input node of a layer must be specified as output of another layer"\
                   "presented before the layer declaration", name);
      int value = static_cast<int>(node_names.size());
      node_name_map[key] = value;
      node_names.push_back(key);
      return value;
    } else {
      return it->second;
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
    utils::Assert(param.num_nodes == static_cast<int>(node_names.size()),
                  "num_nodes is inconsistent with node_names");
    param.init_end = 1;
  }
  /*! \brief clear the configurations */
  inline void ClearConfig(void) {
    defcfg.clear();
    for (size_t i = 0; i < layercfg.size(); ++i) {
      layercfg[i].clear();
    }
  }
};
}  // namespace nnet
}  // namespace cxxnet
#endif
