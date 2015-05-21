/*!
 * \file convert.cpp
 * \brief convert caffe model to cxx model
 * \author Zehua Huang
 */

#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <ctime>
#include <string>
#include <cstring>
#include <vector>
#include <nnet/nnet.h>
#include <utils/config.h>
#include <nnet/neural_net-inl.hpp>
#include <nnet/nnet_impl-inl.hpp>
#include <layer/convolution_layer-inl.hpp>
#include <layer/cudnn_convolution_layer-inl.hpp>
#include <layer/fullc_layer-inl.hpp>

#include <caffe/blob.hpp>
#include <caffe/net.hpp>
#include <caffe/util/io.hpp>
#include <caffe/vision_layers.hpp>

using namespace std;
namespace cxxnet {
  class CaffeConverter {
  public:
    CaffeConverter() {
      this->net_type_ = 0;
    }

    ~CaffeConverter() {
    if (net_trainer_ != NULL) {
      delete net_trainer_;
    }
    }

    void Convert(int argc, char *argv[]) {
      if (argc != 5) {
        printf("Usage: <caffe_proto> <caffe_model> <cxx_config> <cxx_model_output>\n");
        return;
      }

      this->InitCaffe(argv[1], argv[2]);
      this->InitCXX(argv[3]);
      this->TransferNet();
      this->SaveModel(argv[4]);
    }

  private:
    inline void InitCaffe(const char *network_params, const char *network_snapshot) {
      caffe::Caffe::set_mode(caffe::Caffe::CPU);

      caffe_net_.reset(new caffe::Net<float>(network_params, caffe::TEST));
      caffe_net_->CopyTrainedLayersFrom(network_snapshot);
    }

    inline void InitCXX(const char *configPath) {
      utils::ConfigIterator itr(configPath);
      while (itr.Next()) {
        this->SetParam(itr.name(), itr.val());
      }

      net_trainer_ = (nnet::CXXNetThreadTrainer<cpu>*)this->CreateNet();
      net_trainer_->InitModel();
    }

    inline void TransferNet() {
      const vector<caffe::shared_ptr<caffe::Layer<float> > >& caffe_layers = caffe_net_->layers();
      const vector<string> & layer_names = caffe_net_->layer_names();

      for (size_t i = 0; i < layer_names.size(); ++i) {
        if (caffe::InnerProductLayer<float> *caffe_layer = dynamic_cast<caffe::InnerProductLayer<float> *>(caffe_layers[i].get())) {
          printf("Dumping InnerProductLayer %s\n", layer_names[i].c_str());

          vector<caffe::shared_ptr<caffe::Blob<float> > >& blobs = caffe_layer->blobs();
          caffe::Blob<float> &caffe_weight = *blobs[0];
          caffe::Blob<float> &caffe_bias = *blobs[1];

          mshadow::TensorContainer<mshadow::cpu, 2> weight;
          weight.Resize(mshadow::Shape2(caffe_weight.num(), caffe_weight.channels()));
          for (int n = 0; n < caffe_weight.num(); n++) {
            for (int c = 0; c < caffe_weight.channels(); c++) {
              weight[n][c] = caffe_weight.data_at(n, c, 0, 0);
            }
          }

          mshadow::TensorContainer<mshadow::cpu, 2> bias;
          bias.Resize(mshadow::Shape2(caffe_bias.count(), 1));
          for (int b = 0; b < caffe_bias.count(); b++) {
            bias[b] = caffe_bias.data_at(b, 0, 0, 0);
          }

          net_trainer_->SetWeight(weight, layer_names[i].c_str(), "wmat");
          net_trainer_->SetWeight(bias, layer_names[i].c_str(), "bias");

        } else if (caffe::ConvolutionLayer<float> *caffe_layer = dynamic_cast<caffe::ConvolutionLayer<float> *>(caffe_layers[i].get())) {
          printf("Dumping ConvolutionLayer %s\n", layer_names[i].c_str());

          vector<caffe::shared_ptr<caffe::Blob<float> > >& blobs = caffe_layer->blobs();
          caffe::Blob<float> &caffe_weight = *blobs[0];
          caffe::Blob<float> &caffe_bias = *blobs[1];

          mshadow::TensorContainer<mshadow::cpu, 2> weight;
          weight.Resize(mshadow::Shape2(caffe_weight.num(),
              caffe_weight.count() / caffe_weight.num()));

          for (int n = 0; n < caffe_weight.num(); n++) {
            for (int c = 0; c < caffe_weight.channels(); c++) {
              for (int h = 0; h < caffe_weight.height(); h++) {
                for (int w = 0; w < caffe_weight.width(); w++) {
                  float data;
                  if (i==0) {
                    data = caffe_weight.data_at(n, 2 - c, h, w);
                  } else {
                    data = caffe_weight.data_at(n, c, h, w);
                  }

                  weight[n][(c * caffe_weight.height() +
                         h) * caffe_weight.width() +
                         w] = data;
                } // width
              } // height
            } // channel
          } // num

          mshadow::TensorContainer<mshadow::cpu, 2> bias;
          bias.Resize(mshadow::Shape2(caffe_bias.count(), 1));
          for (int b = 0; b < caffe_bias.count(); b++) {
            bias[b] = caffe_bias.data_at(b, 0, 0, 0);
          }

          net_trainer_->SetWeight(weight, layer_names[i].c_str(), "wmat");
          net_trainer_->SetWeight(bias, layer_names[i].c_str(), "bias");

        } else {
          printf("Ignoring layer %s\n", layer_names[i].c_str());
        }
      }
    }

    inline void SaveModel(const char *save_path) {
      dmlc::Stream *fo = dmlc::Stream::Create(save_path, "wb");
      fo->Write(&net_type_, sizeof(int));
      net_trainer_->SaveModel(*fo);
      delete fo;
      printf("Model saved\n");
    }

    inline void SetParam(const char *name , const char *val) {
      cfg_.push_back(std::make_pair(std::string(name), std::string(val)));
    }

    // create a neural net
    inline nnet::INetTrainer* CreateNet(void) {
      nnet::INetTrainer *net = nnet::CreateNet<mshadow::cpu>(net_type_);

      for (size_t i = 0; i < cfg_.size(); ++ i) {
        net->SetParam(cfg_[i].first.c_str(), cfg_[i].second.c_str());
      }
      return net;
    }

  private:
    /*! \brief type of net implementation */
    int net_type_;
    /*! \brief trainer */
    nnet::CXXNetThreadTrainer<cpu> *net_trainer_;
  private:
    /*! \brief all the configurations */
    std::vector<std::pair<std::string, std::string> > cfg_;
    /*! \brief caffe net reference */
    caffe::shared_ptr<caffe::Net<float> > caffe_net_;
  };
}

int main(int argc, char *argv[]) {
  cxxnet::CaffeConverter converter;
  converter.Convert(argc, argv);
  return 0;
}
