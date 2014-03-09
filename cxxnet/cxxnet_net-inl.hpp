#ifndef CXXNET_NET_INL_HPP
#define CXXNET_NET_INL_HPP
#pragma once
/*!
 * \file cxxnet_nn.h
 * \brief implementation of netural nets
 * \author Tianqi Chen, Bing Xu
 */
#include "cxxnet.h"
#include "cxxnet_net.h"

namespace cxxnet {    
    /*! \brief data structure that contains general shape of network */
    struct NetModel{
    public:
        /*! \brief general model parameter */
        struct Param{
            /*! \brief number of nodes in the network */
            int num_nodes;
            /*! \brief number of layers in the network */
            int num_layers;
            /*! \brief input shape, not including batch dimension */
            mshadow::Shape<3> shape_in;
            /*! \brief reserved flag, used to extend data structure */
            int reserved_flag;
            /*! \brief constructor, reserved flag */
            Param( void ){ 
                reserved_flag = 0;
            } 
        };
        /*! \brief information about each layer */
        struct LayerInfo{
            /*! \brief type of layer */
            int type;
            /*! \brief input node index */
            int nindex_in;
            /*! \brief output node index */
            int nindex_out;
        };
    public:
        /*! \brief model parameter */
        Param param;
        /*! \brief information about each layers */        
        std::vector<LayerInfo> layers;
    public:
        /*! \brief set model parameters */
        inline void SetParam( const char *name, const char *val ){
            if( layers.size() != 0 ) return;
            if( !strcmp( name, "input_shape") ){
                unsigned x, y, z;
                utils::Assert( sscanf( val, "%u,%u,%u", &z,&y,&x ) ==3, 
                               "input_shape must be three consecutive integers without space example: 1,1,200 " );
                param.shape_in[0] = x; param.shape_in[1] = y; param.shape_in[2] = y; 
            }
        }
        /*! \brief guess parameters, from current setting */
        inline void InitParams( void ){
            param.num_layers = static_cast<int>( layers.size() );
            for( size_t i = 0; i < layers.size(); ++ i ){
                param.num_nodes = std::max( layers[i].nindex_out + 1, param.num_nodes) ;
            }
        }
    };

    /*! 
     * \brief data structure of netural net  
     * \tparam xpu data storage type
     */
    template<typename xpu>
    struct NeuralNet{
        /*! \brief nodes in the neural net */
        std::vector< Node<xpu> > nodes;
        /*! \brief layers in the neural net */
        std::vector<ILayer*>     layers;
        /*! \brief updaters in the neural net */
        std::vector<IUpdater*>   updaters;
        /*! \brief random number generator */        
        mshadow::Random<xpu>     rnd;
        ~NeuralNet( void ){
        }
        /*! \brief set parameters */
        inline void FreeSpace( void ){
            for( size_t i = 0; i < nodes.size(); ++ i ){
                mshadow::FreeSpace( nodes[i].data );
            } 
            for( size_t i = 0; i < layers.size(); ++ i ){
                delete layers[i];
            }
            for( size_t i = 0; i < updaters.size(); ++ i ){
                delete updaters[i];
            }
            nodes.clear(); layers.clear(); updaters.clear();
        }
        /*! \brief input node */
        inline Node& in( void ){
            return nodes[0];
        }
        /*! \brief output node */
        inline Node& out( void ){
            return nodes[0];
        }
        inline void InitModel( void ) {
        }
        inline void SaveModel( mshadow::utils::IStream &fo ) const {
            
        }
        virtual void LoadModel( mshadow::utils::IStream &fi ) {
        }
        /*! 
         * \brief forward prop 
         * \param is_train whether is training phase
         */
        inline void Forward( bool is_train ){
            for( size_t i = 0; i < layers.size(); ++ i ){
                layers[i]->Forward( is_train );
            }
        }
        /*! \brief backprop */
        inline void Backprop( void ){
            for( size_t i = layers.size(); i > 0; -- i ){
                layers[i-1]->Backprop( i != 1 );
            }
        }
        /*! \brief update model parameters  */
        inline void Update( void ){
            for( size_t i = 0; i < updaters.size(); ++ i ){
                updaters[i]->Update();
            }
        }
    };
    
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
#endif // CXXNET_NET_INL_HPP
