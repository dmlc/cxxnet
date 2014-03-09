#ifndef CXXNET_NN_H
#define CXXNET_NN_H
#pragma once
#include "cxxnet.h"
/*!
 * \file cxxnet_nn.h
 * \brief Build neural network structure
 * \author Tianqi Chen, Bing Xu
 */
#include "cxxnet.h"

namespace cxxnet {
    /*! \brief */
    template<typename xpu>
    class NeuralNet {
    public:
        /*! \brief constuctor for NeuralNet
         *  \param param configure string for the network
         *  \param input input batch node
         *  \param target target batch node
         */
        explicit NeuralNet(const char* param, Node<xpu> &input, Node<xpu> &target) {
            rnd = mshadow::Random<xpu>(1);
            // Init hidden layers
            // set data
            // run
            num_layers_ = layers.size();
        }
        void InitNetwork<cpu>(const char *param) {
            // Get layer info
            // Get node info
            // Node<cpu> in, out;
            // in.data = mshadow::NewCTensor( mshadow::Shape4(h,l,c, batch) , 1.0f );
            // out.data = mshadow::NewCTensor (in.data.shape);
            // ILayer *layer = CreateLayer(type, rnd, in, out);
            // layers.push_back(layer);
        }
        void InitNetwork<gpu>(const char *param) {
            // Get layer info
            // Get node info
            // Node<gpu> in, out;
            // in.data = mshadow::NewGTensor( mshadow::Shape4(h,l,c, batch) , 1.0f );
            // out.data = mshadow::NewGTensor (in.data.shape);
            // ILayer *layer = CreateLayer(type, rnd, in, out);
            // Set param for layer
            // layers.push_back(layer);
            // GetUpdater
            // push back updater
        }
        void Forward(bool is_train) {
            for (int i = 0; i < num_layers_; ++i) {
                layers[i].Forward(is_train);
            }
            // Update Batch
        }

        void Backprop(bool is_firstlayer) {
            for (int i = num_layers_ - 1; i >= 0; --i) {
                layers[i].Backprop(is_firstlayer);
            }
            // for (int i = num_layers_ - 1; i >= 0; --i) {
            //      updaters[i].Update();
            // }
            // Update Bacth
        }

    private:
        // TODO: support like multi-output nn
        std::vector<ILayer*> layers;
        std::vector<IUpdater*> &updaters
        mshadow::Random<xpu> rnd;
        Node<xpu> in_;
        Node<xpu> target_;
        int num_layers_;




    }; // class NeuralNet


}; // namespace cxxnet
#endif // CXXNET_NN_H
