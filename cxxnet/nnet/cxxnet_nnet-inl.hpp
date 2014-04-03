#ifndef CXXNET_NET_INL_HPP
#define CXXNET_NET_INL_HPP
#pragma once
/*!
 * \file cxxnet_net-inl.hpp
 * \brief implementation of netural nets
 * \author Tianqi Chen, Bing Xu
 */
#include "cxxnet_nnet.h"
#include "../core/cxxnet_core.h"
#include "../utils/cxxnet_metric.h"

namespace cxxnet {
    using namespace mshadow::utils;
    using namespace mshadow::expr;

    /*! \brief data structure that contains general shape of network */
    struct NetMetaModel{
    public:
        /*! \brief general model parameter */
        struct Param{
            /*! \brief number of nodes in the network */
            int num_nodes;
            /*! \brief number of layers in the network */
            int num_layers;
            /*! \brief input shape, not including batch dimension */
            mshadow::Shape<3> shape_in;
            /*! \brief whether the network is initialized */
            int init_end;
            /*! \brief number of epoches pass so far */
            int64_t num_epoch_passed;
            /*! \brief reserved flag, used to extend data structure */
            int reserved_flag;
            /*! \brief constructor, reserved flag */
            Param( void ){
                init_end = 0;
                num_epoch_passed = 0;
                reserved_flag = 0;
            }
            /*! \brief get input shape, given number of batches */
            mshadow::Shape<4> GetShapeIn( index_t nbatch ) const{
                if( shape_in[2] == 1 && shape_in[1] == 1 ){
                    return mshadow::Shape4( 1, 1, nbatch, shape_in[0] );
                }else{
                    return mshadow::Shape4( nbatch, shape_in[2], shape_in[1], shape_in[0] );
                }
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
            inline bool operator==( const LayerInfo &b ) const{
                return type == b.type && nindex_in == b.nindex_in && nindex_out == b.nindex_out;
            }
        };
    public:
        /*! \brief model parameter */
        Param param;
        /*! \brief information about each layers */
        std::vector<LayerInfo> layers;
    public:
        /*! \brief set model parameters */
        inline void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "reset_epoch") ) param.num_epoch_passed = (int64_t)atol( val );
            if( param.init_end != 0 ) return;
            if( !strcmp( name, "input_shape") ){
                unsigned x, y, z;
                Assert( sscanf( val, "%u,%u,%u", &z,&y,&x ) ==3,
                               "input_shape must be three consecutive integers without space example: 1,1,200 " );
                param.shape_in[0] = x; param.shape_in[1] = y; param.shape_in[2] = z;
            }
        }
        /*! \brief guess parameters, from current setting */
        inline void InitModel( void ){
            param.num_nodes = 0;
            param.num_layers = static_cast<int>( layers.size() );
            for( size_t i = 0; i < layers.size(); ++ i ){
                param.num_nodes = std::max( layers[i].nindex_out + 1, param.num_nodes );
            }
            param.init_end = 1;
        }
        virtual void SaveModel( mshadow::utils::IStream &fo ) const {
            fo.Write( &param, sizeof( Param ) );
            fo.Write( &layers[0], sizeof(LayerInfo) * layers.size() );
        }
        virtual void LoadModel( mshadow::utils::IStream &fi ) {
            Assert( fi.Read( &param, sizeof( Param ) ) != 0 );
            layers.resize( param.num_layers );
            if( layers.size() != 0 ){
                Assert( fi.Read( &layers[0], sizeof(LayerInfo) * layers.size() ) != 0 );
            }
        }
    };

    /*! \brief helper class to config networks */
    struct NetConfigHelper{
    public:
        NetConfigHelper( NetMetaModel &meta ):meta(meta){
            this->netcfg_mode = 0;
            this->updater_type = "sgd";
            this->batch_size   = 100;
        }
        // set parameters
        inline void SetParam( const char *name, const char *val ){
            meta.SetParam( name, val );
            if( !strcmp( name, "batch_size" ) ) batch_size = atoi( val );
            if( !strcmp( name, "updater" ) )    updater_type = val;
            if( !strcmp( name, "netconfig" ) && !strcmp( val, "start") ) netcfg_mode = 1;
            if( !strcmp( name, "netconfig" ) && !strcmp( val, "end") )   netcfg_mode = 0;

            if( !strncmp( name, "layer[", 6 ) ){
                netcfg_mode = 2;
                if( meta.param.init_end == 0 ){
                    meta.layers.push_back( this->GetLayerInfo( name, val ) );
                    meta.param.num_layers = static_cast<int>( meta.layers.size() );
                }
            }
            if( netcfg_mode == 2 ){
                // layer specific configuration
                netcfg.push_back( std::make_pair( std::string(name), std::string(val) ) );
            }else{
                defcfg.push_back( std::make_pair( std::string(name), std::string(val) ) );
            }
        }
        template<typename xpu>
        inline void ConfigLayers( std::vector< Node<xpu> >& nodes,
                                  std::vector<ILayer*>& layers,
                                  std::vector<IUpdater*>& updaters, bool init_model ){
            // default configuration
            int layer_index = -1;
            for( size_t i = 0; i < netcfg.size(); ++i ){
                const char* name = netcfg[i].first.c_str();
                const char* val  = netcfg[i].second.c_str();
                if( !strncmp( name, "layer[", 6 ) ){
                    ++ layer_index;
                    Assert( layer_index >= 0 && layer_index < meta.param.num_layers );
                    NetMetaModel::LayerInfo inf = this->GetLayerInfo( name, val );
                    Assert( inf == meta.layers[layer_index], "config setting mismatch" );
                    // set global parameters
                    for( size_t j = 0; j < defcfg.size(); ++ j ){
                        layers[ layer_index ]->
                            SetParam( defcfg[j].first.c_str(), defcfg[j].second.c_str() );
                    }
                }else{
                    Assert( layer_index >= 0 );
                    layers[ layer_index ]->SetParam( name, val );
                }
            }
            // adjust node Shape
            nodes[0].data.shape = meta.param.GetShapeIn( batch_size );
            for( size_t i = 0; i < layers.size(); ++i ){
                layers[i]->InitLayer();
                if( init_model ) layers[i]->InitModel();
            }
            // configure updaters
            layer_index = -1;
            size_t ustart = 0;

            for( size_t i = 0; i < netcfg.size(); ++i ){
                const char* name = netcfg[i].first.c_str();
                const char* val  = netcfg[i].second.c_str();
                if( !strncmp( name, "layer[", 6 ) ){
                    ++ layer_index;
                    ustart = updaters.size();
                    layers[ layer_index ]->GetUpdaters( updater_type.c_str(), updaters );
                    for( size_t j = ustart; j < updaters.size(); ++ j ){
                        for( size_t k = 0; k < defcfg.size(); ++ k ){
                            updaters[j]->
                                SetParam( defcfg[k].first.c_str(), defcfg[k].second.c_str() );
                        }
                    }
                }else{
                    Assert( layer_index >= 0 );
                    for( size_t j = ustart; j < updaters.size(); ++ j ){
                        updaters[j]->SetParam( name, val );
                    }
                }
            }
            for( size_t i = 0; i < updaters.size(); ++ i ){
                updaters[i]->Init();
            }
        }
    private:
        inline NetMetaModel::LayerInfo GetLayerInfo( const char *name, const char *val ){
            int a, b;
            char ltype[256],tag[256];
            Assert( sscanf( name, "layer[%d->%d]", &a, &b ) == 2, "invalid config format, correct example: layer[1->2]" );
            Assert( sscanf( val , "%[^:]:%s", ltype, tag ) >= 1, "invalid config format" );
            NetMetaModel::LayerInfo inf;
            inf.nindex_in = a; inf.nindex_out = b;
            inf.type = GetLayerType( ltype );
            return inf;
        }
    private:
        NetMetaModel &meta;
        // type of updater
        std::string updater_type;
        // configures about network
        std::vector< std::pair< std::string, std::string > > netcfg, defcfg;
        // number of batch size
        int batch_size;
        // whether in net config mode
        int netcfg_mode;
    };

    /*!
     * \brief data structure of netural net
     * \tparam xpu data storage type
     */
    template<typename xpu>
    struct NeuralNet{
    public:
        /*!\brief do not print anything */
        int silent;
        /*! \brief meta information about network */
        NetMetaModel meta;
        /*! \brief configure helper */
        NetConfigHelper cfg;
        /*! \brief nodes in the neural net */
        std::vector< Node<xpu> > nodes;
        /*! \brief layers in the neural net */
        std::vector<ILayer*>     layers;
        /*! \brief updaters in the neural net */
        std::vector<IUpdater*>   updaters;
        /*! \brief random number generator */
        mshadow::Random<xpu>     rnd;
        /*! \brief node factory */
        NodeFactory<xpu> nfactory;
    public:
        /*! \brief constructor */
        NeuralNet( void ): cfg(meta),rnd(0){
            silent = 0;
        }
        /*! \brief destructor */
        ~NeuralNet( void ){
            this->FreeSpace();
        }
        /*! \brief input node */
        inline Node<xpu>& in( void ){
            return nodes[0];
        }
        /*! \brief output node */
        inline Node<xpu>& out( void ){
            return nodes.back();
        }
        /*! \brief set parameter */
        inline void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "seed") ) rnd.Seed( atoi( val ) );
            if( !strcmp( name, "silent") ) silent = atoi(val);
            if( !strcmp( name, "memlimit") ) nfactory.SetMemLimit( val );
            cfg.SetParam( name, val );
        }
        /*! \brief intialize model parameters */
        inline void InitModel( void ) {
            this->FreeSpace();
            meta.InitModel();
            for( int i = 0; i < meta.param.num_nodes; ++i ){
                nodes.push_back( nfactory.CreateNode() );
            }
            for( int i = 0; i < meta.param.num_layers; ++ i ){
                Assert( layers.size() == (size_t) i );
                const NetMetaModel::LayerInfo &info = meta.layers[i];
                layers.push_back( CreateLayer( info.type, rnd, nodes[ info.nindex_in ], nodes[ info.nindex_out ] ) );
            }
            cfg.ConfigLayers( nodes, layers, updaters, true );
            this->InitNodes();
        }
        /*! \brief save model to file */
        inline void SaveModel( mshadow::utils::IStream &fo ) const {
            meta.SaveModel( fo );
            for( int i = 0; i < meta.param.num_layers; ++ i ){
                layers[i]->SaveModel( fo );
            }
        }
        /*! \brief load model from stream */
        inline void LoadModel( mshadow::utils::IStream &fi ) {
            this->FreeSpace();
            meta.LoadModel( fi );
            for( int i = 0; i < meta.param.num_nodes; ++i ){
                nodes.push_back( nfactory.CreateNode() );
            }
            for( int i = 0; i < meta.param.num_layers; ++ i ){
                const NetMetaModel::LayerInfo &info = meta.layers[i];
                Assert( layers.size() == (size_t) i );
                layers.push_back ( CreateLayer( info.type, rnd, nodes[ info.nindex_in ], nodes[ info.nindex_out ] ) );
                layers[i]->LoadModel( fi );
            }
            cfg.ConfigLayers( nodes, layers, updaters, false );
            this->InitNodes();
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
                updaters[i]->Update( meta.param.num_epoch_passed );
            }
            // update epoch
            meta.param.num_epoch_passed += 1;
        }
        /*!
         * \brief notify round start
         * \param round round counter
         */
        virtual void StartRound( int round ) {
            for( size_t i = 0; i < updaters.size(); ++ i ){
                updaters[i]->StartRound( round );
            }
        }
    private:
        /*! \brief check the node shapes */
        inline void InitNodes( void ){
            for( size_t i = 0; i < nodes.size(); ++ i ){
                mshadow::Shape<4> s = nodes[i].data.shape;
                nodes[i].Pin(); nodes[i].Unpin();
                if( !silent ){
                    printf("node[%d].shape: %u,%u,%u,%u\n",(int)i, s[3],s[2],s[1],s[0] );
                }
            }
        }
        /*! \brief set parameters */
        inline void FreeSpace( void ){
            for( size_t i = 0; i < nodes.size(); ++ i ){
                nodes[i].FreeSpace();
            }
            for( size_t i = 0; i < layers.size(); ++ i ){
                delete layers[i];
            }
            for( size_t i = 0; i < updaters.size(); ++ i ){
                delete updaters[i];
            }
            nodes.clear(); layers.clear(); updaters.clear();
        }
    };

    /*! \brief implementation of neural network trainer */
    template<typename xpu>
    class CXXNetTrainer : public INetTrainer{
    public:
        CXXNetTrainer( int devid ){
            loss_type = 0; round = 0;
            printf("CXXNetTrainer, devCPU=%d\n", xpu::kDevCPU );
            if(! xpu::kDevCPU ){
                mshadow::InitTensorEngine( devid );
            }
        }
        virtual ~CXXNetTrainer( void ){
            if(! xpu::kDevCPU ){
                mshadow::ShutdownTensorEngine();
            }
        }
        virtual void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "loss" ) )  loss_type = atoi( val );
            if( !strcmp( name, "top_n") ) top_n_ = atoi(val);
            if( !strcmp( name, "metric") ) metric.AddMetric( val );
            net.SetParam( name, val );
        }
        virtual void InitModel( void ) {
            net.InitModel();
        }
        virtual void SaveModel( mshadow::utils::IStream &fo ) const {
            net.SaveModel( fo );
        }
        virtual void LoadModel( mshadow::utils::IStream &fi ) {
            net.LoadModel( fi );
        }
        // tell trainer which round it is
        virtual void StartRound( int round ) {
            net.StartRound( round );
            this->round = round;
        }
        virtual void Update ( const DataBatch& batch ) {
            net.in().Pin();
            mshadow::Copy( net.in().data, batch.data );
            net.in().Unpin();

            net.Forward( true );
            this->SyncOuput();

            net.out().Pin();
            mshadow::Copy( net.out().data[0][0], temp );
            net.out().Unpin();

            this->SetLoss( batch.labels );
            net.Backprop();
            net.Update();
        }
        virtual void Evaluate( FILE *fo, IIterator<DataBatch> *iter_eval, const char* evname ){
            metric.Clear();
            iter_eval->BeforeFirst();
            while( iter_eval->Next() ){
                const DataBatch& batch = iter_eval->Value();
                std::vector<float> preds;
                if (loss_type == 3 && top_n_ > 0) {
                    this->Predict(preds, batch, true)
                    metric.AddEval( &preds[0], batch.labels, preds.size(), top_n_);
                } else {
                    this->Predict( preds, batch );
                    metric.AddEval( &preds[0], batch.labels, preds.size() );
                }
            }
            metric.Print( fo, evname );
        }
        virtual void Predict( std::vector<float> &preds, const DataBatch& batch ) {
            this->PreparePredTemp( batch );
            for( index_t i = 0; i <temp.shape[1]; ++i ){
                preds.push_back( this->TransformPred( temp[i] ) );
            }
        }
        virtual void Predict(std::vector<float> &preds, const DataBatch& batch, bool is_rank) {
            this->PreparePredTemp( batch );
            for( index_t i = 0; i <temp.shape[1]; ++i ){
                this->TransformPred( temp[i] );
                for (int j = 0; i < top_k; ++j) {
                    preds.push_back(tmp_index_[j]);
                }
            }
        }
    protected:
        // put prediction into temp
        virtual void PreparePredTemp( const DataBatch& batch ){
            net.in().Pin();
            mshadow::Copy( net.in().data, batch.data );
            net.in().Unpin();
            net.Forward( false );
            this->SyncOuput();
        }
    private:
        inline void SyncOuput( void ){
            mshadow::Shape<4> oshape  = net.out().data.shape;
            Assert( net.out().is_mat() );
            temp.Resize( mshadow::Shape2( oshape[1], oshape[0] ) );
            net.out().Pin();
            mshadow::Copy( temp, net.out().data[0][0] );
            net.out().Unpin();
        }
        inline float TransformPred( mshadow::Tensor<cpu,1> pred ){
            switch( loss_type ){
            case 0: return GetMaxIndex( pred );
            case 1: return pred[0];
            case 2: return 1.0f/(1.0f+std::exp(-pred[0]));
            case 3: return MakeRankPred(pred);
            default: Error("unknown loss type"); return 0.0f;
            }
        }
        inline void SetLoss( mshadow::Tensor<cpu,1> pred, float label ){
            switch( loss_type ){
            case 0:
            case 3:{
                index_t k = static_cast<index_t>(label);
                utils::Assert( k < pred.shape[0], "label exceed output bound" );
                pred[ k ] -= 1.0f; break;
            }
            case 1: pred[ 0 ] -=  label; break;
            case 2: pred[ 0 ] = 1.0f/(1.0f+std::exp(-pred[0])) - label; break;
            default: Error("unknown loss type");
            }
        }
        inline void SetLoss( const float* labels ){
            if( loss_type == 1 || loss_type == 2 ){
                Assert( temp.shape[0] == 1, "regression can only have 1 output size" );
            }
            for( index_t i = 0; i <temp.shape[1]; ++i ){
                this->SetLoss( temp[i], labels[i] );
            }
            net.out().Pin();
            mshadow::Copy( net.out().data[0][0], temp );
            net.out().Unpin();
        }
        inline static int GetMaxIndex( mshadow::Tensor<cpu,1> pred ){
            index_t maxidx = 0;
            for( index_t i = 1; i < pred.shape[0]; ++ i ){
                if( pred[i] > pred[maxidx] ) maxidx = i;
            }
            return maxidx;
        }
        inline static float MakeRankPred(mshadow::Tensor<cpu, 1> pred) {
            std::priority_queue<std::pair<float, int>, \
                std::vector<std::pair<float, int> >, \
                utils::PairCompare<float, int> > pq;
            tmp_index_.clear();
            tmp_index_.resize(top_n_);
            for (index_t i = 0; i < pred.shape[0]; ++i) {
                pq.push(std::make_pair(pred[i], i));
            }
            // utils::Assert(pq.size() > top_n);
            for (int i = 0; i < top_n_; ++i) {
                tmp_index_[i] = pq.top().second;
                pq.pop();
            }
            return 0.0f;
        }
    protected:
        // get top n predict
        int top_n_;
        // current round
        int round;
        // loss function
        int loss_type;
        // evaluator
        utils::MetricSet metric;
        // temp space
        mshadow::TensorContainer<cpu,2> temp;
        // true net
        NeuralNet<xpu> net;
        // tmp stoage of top index
        vector<index_t> tmp_index_;
    }; // class NeuralNet


    /*!
     * \brief implementation of averaging neural network trainer
     *        will perform weight averaging during predictions
     */
    template<typename xpu>
    class CXXAvgNetTrainer: public CXXNetTrainer<xpu>{
    public:
        CXXAvgNetTrainer( int devid ) : CXXNetTrainer<xpu>(devid){
            num_burn = INT_MAX;
            num_avg_record = 0;
        }
        virtual ~CXXAvgNetTrainer( void ){}
        virtual void SetParam( const char *name, const char *val ){
            CXXNetTrainer<xpu>::SetParam( name, val );
            if( !strcmp( "num_inst",name) ) num_avg_record = atoi(val);
            if( !strcmp( "num_burn",name) ) num_burn = atoi(val);
        }
        virtual void InitModel( void ){
            CXXNetTrainer<xpu>::InitModel();
            ref_counter.resize( num_avg_record, 0 );
            mshadow::Shape<2> s = this->net.out().data[0][0].shape;
            avg_pred.Resize( mshadow::Shape2( num_avg_record, s[0] ), 0.0f );
        }
        virtual void SaveModel( mshadow::utils::IStream &fo ) const {
            CXXNetTrainer<xpu>::SaveModel( fo );
            fo.Write( &num_avg_record, sizeof(int) );
            fo.Write( &ref_counter[0], ref_counter.size() * sizeof(int) );
            avg_pred.SaveBinary( fo );
        }
        virtual void LoadModel( mshadow::utils::IStream &fi ) {
            CXXNetTrainer<xpu>::LoadModel( fi );
            Assert( fi.Read( &num_avg_record, sizeof(int) )!= 0 );
            ref_counter.resize( num_avg_record );
            Assert( fi.Read( &ref_counter[0], ref_counter.size() * sizeof(int) ) != 0 );
            avg_pred.LoadBinary( fi );
        }
    protected:
        virtual void PreparePredTemp( const DataBatch& batch ){
            CXXNetTrainer<xpu>::PreparePredTemp( batch );
            mshadow::Tensor<cpu,2> &temp = this->temp;
            Assert( batch.inst_index != NULL, "CXXAvgNetTrainer need inst_index" );
            for( index_t i = 0; i < temp.shape[1]; ++i ){
                unsigned ridx = batch.inst_index[ i ];
                Assert( ridx < num_avg_record, "inst_index exceed num_avg_record" );
                if( ref_counter[ ridx ] > this->round ) continue;
                ref_counter[ ridx ] = this->round + 1;
                int diff = this->round - num_burn;
                if( diff < 1 ) diff = 1;
                float alpha = 1.0f / diff;
                avg_pred[ridx] = (1.0f-alpha) * avg_pred[ridx] + alpha*temp[i];
                mshadow::Copy( temp[ i ], avg_pred[ridx] );
            }
        }
    private:
        // number of burn in rounds, start averagin after this
        int num_burn;
        // number of records to do averaging
        unsigned num_avg_record;
        // reference counter
        std::vector<int> ref_counter;
        // average prediction
        mshadow::TensorContainer<cpu,2> avg_pred;
    };
}; // namespace cxxnet

namespace cxxnet{
    template<typename xpu>
    INetTrainer* CreateNet_( int net_type, int devid = -1 ){
        switch( net_type ){
        case 0: return new CXXNetTrainer<xpu>( devid );
        case 1: return new CXXAvgNetTrainer<xpu>( devid );
        default: Error("unknown net type");
        }
        return NULL;
    }
}; // namespace cxxnet
#endif // CXXNET_NET_INL_HPP
