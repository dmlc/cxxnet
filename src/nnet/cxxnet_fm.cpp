/*!
 * \file cxxnet_net_fm.cpp
 * \brief implementation of factorization machine, do it for fun
 * \author Tianqi Chen, Bing Xu
 */
#include "cxxnet_nnet.h"
#include "../core/cxxnet_core.h"
#include "../utils/cxxnet_utils.h"
#include "../utils/cxxnet_metric.h"

namespace cxxnet{
    using namespace mshadow::utils;
    using namespace mshadow::expr;

    class CXXNetFMTrainer: public INetTrainer{
    public:
        CXXNetFMTrainer(void):rnd_cpu(0){
        }
        virtual ~CXXNetFMTrainer(void){
        }
        virtual void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "input_shape") ){
                unsigned x, y, z;
                Assert( sscanf( val, "%u,%u,%u", &z,&y,&x ) ==3,
                               "input_shape must be three consecutive integers without space example: 1,1,200 " );
                mparam.shape_in[0] = x; mparam.shape_in[1] = y; mparam.shape_in[2] = z;
                utils::Assert( y == 1 && z == 1, "sparse net only accept vector as input" );
                return;
            }
            if( !strcmp( name, "num_factor") ) mparam.num_factor = atoi(val);
            if( !strcmp( name, "init_sigma") ) mparam.init_sigma = (float)atof(val);
            if( !strcmp( name, "base_score") ) mparam.base_score = -logf(1.0f/(float)atof(val) - 1.0f);
            if( !strcmp( name, "metric") ) {
                metric.AddMetric( val ); train_metric.AddMetric( val );
            }
            if( !strcmp( name, "seed") ) rnd_cpu.Seed( atoi(val) );
            if( !strcmp( name, "nthread") ){
                mparam.nthread = atoi(val);
                omp_set_num_threads(mparam.nthread);
            }
            tparam.SetParam(name, val);
        }
        virtual void InitModel( void ){
            Wsp.Resize( mshadow::Shape2( mparam.shape_in[0], mparam.num_factor ) );
            bias.Resize( mshadow::Shape1( mparam.shape_in[0] ) );
            rnd_cpu.SampleGaussian( Wsp, 0.0f, mparam.init_sigma );
            bias = 0.0f;
            if( tparam.silent == 0 ){
                printf("CXXNetFMTrainer: init with %ux%d weight matrix\n", Wsp.shape[1], Wsp.shape[0]);
                printf("FMSGDUpdater: eta=%f, init_sigma=%f\n", tparam.learning_rate, mparam.init_sigma );
            }
        }
        virtual void SaveModel( mshadow::utils::IStream &fo ) const {
            fo.Write( &mparam, sizeof(ModelParam) );
            Wsp.SaveBinary( fo );
            bias.SaveBinary( fo );
        }
        virtual void LoadModel( mshadow::utils::IStream &fi ) {
            utils::Assert( fi.Read( &mparam, sizeof(ModelParam) ) != 0, "Model" );
            Wsp.LoadBinary( fi );
            bias.LoadBinary( fi );
        }
        virtual void Update ( const DataBatch& batch ) {
            temp.Resize( mshadow::Shape2(batch.batch_size, 1) );
            // maybe put a OpenMP here           
            #pragma omp parallal
            {
                // thread local variable
                mshadow::TensorContainer<cpu,1> tmp(mshadow::Shape1(Wsp.shape[0]));
                #pragma omp for schedule(static)                
                for( index_t i = 0; i < batch.batch_size; ++ i ){            
                    const SparseInst inst = batch.GetRowSparse(i);
                    float pred = this->Pred(inst, tmp);
                    this->Update( batch.GetRowSparse(i), tmp, pred );
                    temp[i][0] = pred;                    
                }
            }
            if( tparam.eval_train != 0){
                train_metric.AddEval( temp, batch.labels );
            }
        } 
        virtual void StartRound( int round ){}
        virtual std::string Evaluate( IIterator<DataBatch> *iter_eval, const char* evname ) {
            std::string res;            
            if (tparam.eval_train != 0 ) {
                res += train_metric.Print("train");
                train_metric.Clear();
            }
            if( iter_eval == NULL ) return res;
            metric.Clear();
            iter_eval->BeforeFirst();
            while( iter_eval->Next() ){
                const DataBatch& batch = iter_eval->Value();
                this->PreparePredTemp( batch );
                metric.AddEval( temp.Slice(0, temp.shape[1]-batch.num_batch_padd), batch.labels );
            }
            res += metric.Print( evname );
            return res;       
        }
        virtual void Predict( std::vector<float> &preds, const DataBatch& batch ){
            preds.resize( batch.batch_size );
            this->PreparePredTemp( batch );
            for( index_t i = 0; i < batch.batch_size; ++ i ){
                preds[i] = temp[i][0];
            }            
        }
        /*! \brief inference feature */
        virtual void Inference(int stop_layer, const DataBatch& batch, long total_length, int &header_flag, mshadow::utils::IStream &fo){
            utils::Error("inference is not implemented");
        }       
    private:
        inline void PreparePredTemp(const DataBatch &batch){
            temp.Resize( mshadow::Shape2(batch.batch_size, 1) );
            #pragma omp parallal
            {
                // thread local variable
                mshadow::TensorContainer<cpu,1> tmp(mshadow::Shape1(Wsp.shape[0]));
                #pragma omp for schedule(static)                
                for( index_t i = 0; i < batch.batch_size; ++ i ){            
                    temp[i][0] = this->Pred( batch.GetRowSparse(i), tmp );
                }
            }
        }
        inline float Pred( const SparseInst &inst, mshadow::Tensor<cpu,1> tmp ){
            tmp = 0.0f;
            // make tmp
            double sum = mparam.base_score, sfac = 0.0;            
            for( unsigned j = 0; j < inst.length; ++ j ){
                const SparseInst::Entry &e = inst[j];
                utils::Assert( e.findex < Wsp.shape[1], " feature index exceed bound" );
                tmp += e.fvalue * Wsp[ e.findex ];
                sum += e.fvalue * bias[ e.findex ];
                sfac -= e.fvalue * e.fvalue * mshadow::VDot( Wsp[e.findex], Wsp[e.findex] );
            }
            sfac += mshadow::VDot( tmp, tmp );
            
            return sum + sfac;
        }
        inline void Update( const SparseInst &inst, mshadow::Tensor<cpu,1> tmp, float pred){
            float py = 1.0f/(1.0f+expf(-pred));
            float err = inst.label - py;
            for( unsigned j = 0; j < inst.length; ++ j ){
                const SparseInst::Entry &e = inst[j];
                Wsp[e.findex] += tparam.learning_rate * err * (tmp - Wsp[e.findex]*e.fvalue);
                bias[e.findex] += tparam.learning_rate * err;
                Wsp[e.findex] *= (1.0f - tparam.learning_rate * tparam.wd);
                bias[e.findex] *= (1.0f - tparam.learning_rate * tparam.wd_bias);
            }
        }
    private:
        /*! \brief training parameters */
        struct TrainParam{
            /*! \brief learning rate of the model */
            float learning_rate;
            /*! \brief weight decay of the model */
            float wd;
            /*! \brief weight decay on bias */
            float wd_bias;
            /*! \brief shut up */
            int silent;
            // evaluate train
            int eval_train;
            // constructor
            TrainParam(void){
                learning_rate = 0.001f;
                wd_bias = wd = 0.0f;
                silent = 0;
                eval_train = 1;
            }
            inline void SetParam( const char *name, const char *val ){
                if( !strcmp( name, "learning_rate") ) learning_rate = (float)atof(val);
                if( !strcmp( name, "eta") ) learning_rate = (float)atof(val);
                if( !strcmp( name, "wd_bias") ) wd_bias = (float)atof(val);
                if( !strcmp( name, "wd") ) wd = (float)atof(val);
                if( !strcmp( name, "silent") ) silent = atoi(val);
                if( !strcmp( name, "eval_train") ) eval_train = atoi(val);
            }
        };
        /*! \brief additional input for sparse net */
        struct ModelParam{
            /*! \brief base prediction score */
            float base_score;
            /*! \brief intialize gaussian std for sparse layer */
            float init_sigma;
            /*! \brief number of hidden nodes in the first sparse layer */
            int num_factor;
            /*! \brief input shape, not including batch dimension */
            mshadow::Shape<3> shape_in;
            /*! \brief number of threads used */
            int nthread;
            /*! \brief reserved field */
            int reserved[32];
            /*! \brief default constructor */
            ModelParam(void){
                nthread = 1;
                omp_set_num_threads(nthread);
                base_score = 0.0f;
                shape_in = mshadow::Shape3( 1, 1, 0 );
                init_sigma = 0.0005f;
                num_factor = 64;
                memset( reserved, 0, sizeof(reserved) );
            }            
        };
    private:
        /*! \brief reserved  cpu random number generator */
        mshadow::Random<cpu>     rnd_cpu;        
        /*! \brief evaluator */
        utils::MetricSet metric, train_metric;
        /*! \brief training parameters */
        TrainParam tparam;
        /*! \brief neural net parameter */
        ModelParam mparam;
        /*! \brief temp hidden node for the model */
        mshadow::TensorContainer<cpu,2> Wsp;
        mshadow::TensorContainer<cpu,1> bias;
        mshadow::TensorContainer<cpu,2> temp;
    };    
};

namespace cxxnet {
    INetTrainer* CreateNetCPU( int net_type ){
        return new CXXNetFMTrainer();
    }
    INetTrainer* CreateNetGPU( int net_type ){
        return new CXXNetFMTrainer();
    }    
};

