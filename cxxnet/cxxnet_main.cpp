#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <ctime>
#include <string>
#include <cstring>
#include <vector>
#include "nnet/cxxnet.h"
#include "io/cxxnet_data.h"
#include "utils/cxxnet_config.h"

namespace cxxnet{
    class CXXNetLearnTask{
    public:
        CXXNetLearnTask( void ){
            this->task = "train";
            this->net_type = 0;
            this->net_trainer = NULL;
            this->itr_train = NULL;
            name_model_dir = "models";
            device = "gpu";
            num_round = 10; 
            silent = start_counter = 0; 
            max_round = INT_MAX;
            continue_training = 0;            
            save_period = 1;            
            name_model_in = "NULL";
            name_pred     = "pred.txt";
            print_step    = 100;
            eval_train = 0;
        }
        ~CXXNetLearnTask( void ){
            if( net_trainer != NULL ) delete net_trainer;
            if( itr_train != NULL )   delete itr_train;
            for( size_t i = 0; i < itr_evals.size(); ++ i ){
                delete itr_evals[i];
            }
        }
    public:
        inline int Run( int argc, char *argv[] ){
            if( argc < 2 ){
                printf("Usage: <config>\n"); 
                return 0;
            }
            utils::ConfigIterator itr( argv[1] );
            while( itr.Next() ){
                this->SetParam( itr.name(), itr.val() );
            }
            for( int i = 2; i < argc; i ++ ){
                char name[256], val[256];
                if( sscanf( argv[i], "%[^=]=%s", name, val ) == 2 ){
                    this->SetParam( name, val );
                }
            }            
            this->Init();
            if( !silent ){
                printf("initializing end, start working\n");
            }                
            if( task == "train" ) this->TaskTrain();
            return 0;
        }
        
        inline void SetParam( const char *name , const char *val ){
            if( !strcmp( val, "default") ) return;
            if( !strcmp( name,"net_type"))            net_type = atoi( val );
            if( !strcmp( name,"eval_train"))          eval_train = atoi( val );  
            if( !strcmp( name,"print_step"))          print_step = atoi( val ); 
            if( !strcmp( name,"continue"))            continue_training = atoi( val ); 
            if( !strcmp( name,"save_model" ) )        save_period = atoi( val );
            if( !strcmp( name,"start_counter" ))      start_counter = atoi( val );
            if( !strcmp( name,"model_in" ))           name_model_in = val;
            if( !strcmp( name,"model_dir" ))          name_model_dir= val; 
            if( !strcmp( name,"num_round"  ))         num_round     = atoi( val ); 
            if( !strcmp( name,"max_round"))           max_round = atoi( val ); 
            if( !strcmp( name, "silent") )            silent        = atoi( val );            
            if( !strcmp( name, "pred" ))              name_pred   = val;
            if( !strcmp( name, "task") )              task = val;
            if( !strcmp( name, "dev") )               device = val;
            cfg.push_back( std::make_pair( std::string(name), std::string(val) ) );
        }
    private:
        // configure trainer
        inline void Init( void ){
            if( continue_training == 0 || SyncLastestModel() == 0 ){
                continue_training = 0; 
                net_trainer = this->CreateNet();                
                if( name_model_in == "NULL" ){
                    net_trainer->InitModel();
                }else{
                    this->LoadModel(); 
                }
            }
            this->CreateIterators();
        }
        // load in latest model from model_folder
        inline int SyncLastestModel( void ){
            FILE *fi = NULL, *last = NULL;         
            char name[ 256 ];
            int s_counter = start_counter;
            do{
                if( last != NULL ) fclose( last );
                last = fi;
                sprintf(name,"%s/%04d.model", name_model_dir.c_str(), s_counter ++ );
                fi = fopen64( name, "rb");                
            }while( fi != NULL ); 
            
            if( last != NULL ){
                utils::Assert( fread( &net_type, sizeof(int), 1, last ) > 0, "loading model" );
                net_trainer = this->CreateNet();
                mshadow::utils::FileStream fs( last );
                net_trainer->LoadModel( fs );
                start_counter = s_counter - 1;
                fclose( last );
                return 1;
            }else{
                return 0;
            }
        }
        // load model from file 
        inline void LoadModel( void ){
            FILE *fi = utils::FopenCheck( name_model_in.c_str(), "rb" );
            utils::Assert( fread( &net_type, sizeof(int), 1, fi ) > 0, "loading model" );
            net_trainer = this->CreateNet();
            mshadow::utils::FileStream fs( fi );
            net_trainer->LoadModel( fs );
            fclose( fi );
        }
        // save model into file 
        inline void SaveModel( void ){
            char name[256];
            sprintf(name,"%s/%04d.model" , name_model_dir.c_str(), start_counter ++ );
            if( save_period == 0 || start_counter % save_period != 0 ) return;
            FILE *fo  = utils::FopenCheck( name, "wb" );
            fwrite( &net_type, sizeof(int), 1, fo );
            mshadow::utils::FileStream fs( fo );
            net_trainer->SaveModel( fs );
            fclose( fo );
        }
        // create a neural net
        inline INetTrainer* CreateNet( void ) const{
            INetTrainer* net = cxxnet::CreateNet( net_type, device.c_str() );
            for( size_t i = 0; i < cfg.size(); ++ i ){
                net->SetParam( cfg[i].first.c_str(), cfg[i].second.c_str() );
            }
            return net;
        }
        // iterators
        inline void CreateIterators( void ){
            int flag = 0;
            std::string evname;
            std::vector< std::pair< std::string, std::string> > itcfg;
            for( size_t i = 0; i < cfg.size(); ++ i ){
                const char *name = cfg[i].first.c_str();
                const char *val  = cfg[i].second.c_str();
                if( !strcmp( name, "data" ) ){
                    flag = 1; continue;
                }
                if( !strcmp( name, "eval" ) ){
                    evname = std::string( val );
                    flag = 2; continue;
                }
                if( !strcmp( name, "iter" ) && !strcmp( val, "end" ) ){
                    utils::Assert( flag != 0, "wrong configuration file" );
                    if( flag == 1 ){
                        utils::Assert( itr_train == NULL, "can only have one data" );
                        itr_train = cxxnet::CreateIterator( itcfg );
                    }
                    if( flag == 2 ){
                        itr_evals.push_back( cxxnet::CreateIterator( itcfg ) );
                        eval_names.push_back( evname );
                    }
                    flag = 0; itcfg.clear();
                }               
                if( flag == 0 ) continue;
                itcfg.push_back( cfg[i] );
            }
        }
    private:
        inline void TaskTrain( void ){            
            time_t start    = time( NULL );
            unsigned long elapsed = 0;            
            if( continue_training == 0 ){
                this->SaveModel();
            }
            
            int cc = max_round; 
            while( start_counter <= num_round && cc -- ) {
                if( !silent ){
                    printf("update round %d", start_counter -1 ); fflush( stdout );
                }
                int sample_counter = 0;
                net_trainer->StartRound( start_counter );
                itr_train->BeforeFirst();

                while( itr_train->Next() ){
                    net_trainer->Update( itr_train->Value() );
                    if( ++ sample_counter  % print_step == 0 ){
                        elapsed = (long)(time(NULL) - start); 
                        if( !silent ){
                            printf("\r                                                               \r");
                            printf("round %8d:[%8d] %ld sec elapsed", start_counter-1,
                                   sample_counter, elapsed );
                            fflush( stdout );
                        }
                    }
                }
                {// code handling evaluation
                    fprintf(stderr, "[%d]", start_counter );
                    if( eval_train ){
                        net_trainer->Evaluate( stderr, itr_train, "train" );
                    } 
                    for( size_t i = 0; i < itr_evals.size(); ++i ){
                        net_trainer->Evaluate( stderr, itr_evals[i], eval_names[i].c_str() );
                    }
                    fprintf(stderr, "\n" );
                }
                elapsed = (unsigned long)(time(NULL) - start); 
                this->SaveModel();
            }
            
            if( !silent ){
                printf("\nupdating end, %lu sec in all\n", elapsed );
            }
        }
    private:
        // type of net implementation
        int net_type;
        // trainer 
        INetTrainer *net_trainer;
        // training iterator 
        IIterator<DataBatch>* itr_train;
        // validation iterators
        std::vector< IIterator<DataBatch>* > itr_evals;
        // evaluation names 
        std::vector<std::string> eval_names;
    private:
        // all the configurations
        std::vector< std::pair< std::string, std::string> > cfg;
    private:
        // whether evaluate training loss
        int eval_train;
        // how may samples before print information
        int print_step;
        // number of round to train
        int num_round;
        // maximum number of round to train
        int max_round;
        // continue from model folder
        int continue_training;        
        // whether to save model after each round        
        int save_period;
        // start counter of model
        int start_counter; 
        // whether to be silent
        int silent;
        // device of the trainer
        std::string device;
        // task of the job
        std::string task;
        // input model name 
        std::string name_model_in;
        // training data 
        std::string name_data;
        // folder name of output  
        std::string name_model_dir;
        // file name to write prediction
        std::string name_pred;
    };
};

int main( int argc, char *argv[] ){    
    cxxnet::CXXNetLearnTask tsk;
    return tsk.Run( argc, argv );
}
