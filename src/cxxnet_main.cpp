#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <ctime>
#include <string>
#include <cstring>
#include <vector>
#include <climits>
#include "nnet/nnet.h"
#include "io/data.h"
#include "utils/config.h"

namespace cxxnet{


class CXXNetLearnTask {
 public:
  CXXNetLearnTask(void) {
    this->task = "train";
    this->net_type = 0;
    this->net_trainer = NULL;
    this->itr_train = NULL;
    this->itr_pred  = NULL;
    name_model_dir = "models";
    num_round = 10;
    test_io = 0;
    silent = start_counter = 0;
    max_round = INT_MAX;
    continue_training = 0;
    save_period = 1;
    name_model_in = "NULL";
    name_pred     = "pred.txt";
    print_step    = 100;
    reset_net_type = -1;
    extract_node_name = "";
    output_format = 1;
#if MSHADOW_USE_CUDA
    this->SetParam("dev", "gpu");
#else
    this->SetParam("dev", "cpu");
#endif
  }
  ~CXXNetLearnTask(void) {
    if (net_trainer != NULL) {
      delete net_trainer;
      // shut down tensor engine if it is GPU based
      //if (device == "gpu") mshadow::ShutdownTensorEngine();
    }

    if (itr_train != NULL)   delete itr_train;
    if (itr_pred  != NULL)   delete itr_pred;
    for (size_t i = 0; i < itr_evals.size(); ++ i) {
      delete itr_evals[i];
    }
  }
 public:
  inline int Run(int argc, char *argv[]) {
    if (argc < 2) {
      printf("Usage: <config>\n");
      return 0;
    }

    utils::ConfigIterator itr(argv[1]);
    while (itr.Next()) {
      this->SetParam(itr.name(), itr.val());
    }
    for (int i = 2; i < argc; i ++) {
      char name[256], val[256];
      if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
        this->SetParam(name, val);
      }
    }
    this->Init();
    if (!silent) {
      printf("initializing end, start working\n");
    }
    if (task == "train" || task == "finetune") this->TaskTrain();
    if (task == "pred")   this->TaskPredict();
    if (task == "extract") this->TaskExtractFeature();
    return 0;
  }

  inline void SetParam(const char *name , const char *val) {
    if (!strcmp(val, "default")) return;
    if (!strcmp(name,"net_type"))            net_type = atoi(val);
    if (!strcmp(name,"reset_net_type"))      reset_net_type = atoi(val);
    if (!strcmp(name,"print_step"))          print_step = atoi(val);
    if (!strcmp(name,"continue"))            continue_training = atoi(val);
    if (!strcmp(name,"save_model"))        save_period = atoi(val);
    if (!strcmp(name,"start_counter"))      start_counter = atoi(val);
    if (!strcmp(name,"model_in"))           name_model_in = val;
    if (!strcmp(name,"model_dir"))          name_model_dir= val;
    if (!strcmp(name,"num_round" ))         num_round     = atoi(val);
    if (!strcmp(name,"max_round"))           max_round = atoi(val);
    if (!strcmp(name, "silent"))            silent        = atoi(val);
    if (!strcmp(name, "task"))              task = val;
    if (!strcmp(name, "dev"))               device = val;
    if (!strcmp(name, "test_io"))           test_io = atoi(val);
    if (!strcmp(name, "extract_node_name"))         extract_node_name = val;
    if (!strcmp(name, "output_format")) {
      if  (!strcmp(val, "txt")) output_format = 1;
      else output_format = 0;
    }
    cfg.push_back(std::make_pair(std::string(name), std::string(val)));
  }
 private:
  // configure trainer
  inline void Init(void) {
    if (task == "train" && continue_training) {
      if (SyncLastestModel()) {
        utils::Error("Init: Cannot find models for continue training. \
          Please specify it by model_in instead.");
      } else {
        printf("Init: Continue training from %d\n", start_counter);
        return;
      }
    }
    continue_training = 0;
    if (name_model_in == "NULL") {
      utils::Assert(task == "train", "must specify model_in if not training");
      net_trainer = this->CreateNet();
      net_trainer->InitModel();
    } else {
      if (task == "finetune") {
        this->CopyModel();
      } else {
        this->LoadModel();
      }
    }
  
    this->CreateIterators();
  }
  // load in latest model from model_folder
  inline int SyncLastestModel(void) {
    FILE *fi = NULL, *last = NULL;
    char name[ 256 ];
    int s_counter = start_counter;
    do{
      if (last != NULL) fclose(last);
      last = fi;
      sprintf(name,"%s/%04d.model", name_model_dir.c_str(), s_counter ++);
      fi = fopen64(name, "rb");
    }while (fi != NULL);

    if (last != NULL) {
      utils::Assert(fread(&net_type, sizeof(int), 1, last) > 0, "loading model");
      net_trainer = this->CreateNet();
      utils::FileStream fs(last);
      net_trainer->LoadModel(fs);
      start_counter = s_counter - 1;
      fclose(last);
      return 1;
    }else{
      return 0;
    }
  }
  // load model from file
  inline void LoadModel(void) {
    const char* pos = strrchr(name_model_in.c_str(), '/');
    if (pos != NULL && sscanf(pos + 1, "%d", &start_counter) != 1){
      printf("WARNING: Cannot infer start_counter from model name. Specify it in config if needed\n");
    }
    FILE *fi = utils::FopenCheck(name_model_in.c_str(), "rb");
    utils::Assert(fread(&net_type, sizeof(int), 1, fi) > 0, "loading model");
    net_trainer = this->CreateNet();
    utils::FileStream fs(fi);
    net_trainer->LoadModel(fs);
    fclose(fi);
    ++start_counter;
  }
  // save model into file
  inline void SaveModel(void) {
    char name[256];
    sprintf(name,"%s/%04d.model" , name_model_dir.c_str(), start_counter ++);
    if (save_period == 0 || start_counter % save_period != 0) return;
    FILE *fo  = utils::FopenCheck(name, "wb");
    fwrite(&net_type, sizeof(int), 1, fo);
    utils::FileStream fs(fo);
    net_trainer->SaveModel(fs);
    fclose(fo);
  }
  // create a neural net
  inline nnet::INetTrainer* CreateNet(void) {
    if (reset_net_type != -1) {
      net_type = reset_net_type;
    }
    nnet::INetTrainer *net;
    if (!strncmp(device.c_str(), "gpu", 3)) {
#if MSHADOW_USE_CUDA
      net = nnet::CreateNet<mshadow::gpu>(net_type);
#else
      net = NULL;
      utils::Error("MSHADOW_USE_CUDA was not enabled");
#endif
    } else {
      net = nnet::CreateNet<mshadow::cpu>(net_type);
    }

    for (size_t i = 0; i < cfg.size(); ++ i) {
      net->SetParam(cfg[i].first.c_str(), cfg[i].second.c_str());
    }
    return net;
  }
  inline void InitIter(IIterator<DataBatch>* itr,
                        const std::vector< std::pair< std::string, std::string> > &defcfg) {
    for (size_t i = 0; i < defcfg.size(); ++ i) {
      itr->SetParam(defcfg[i].first.c_str(), defcfg[i].second.c_str());
    }
    itr->Init();

  }
  // iterators
  inline void CreateIterators(void) {
    int flag = 0;
    std::string evname;
    std::vector< std::pair< std::string, std::string> > itcfg;
    std::vector< std::pair< std::string, std::string> > defcfg;
    for (size_t i = 0; i < cfg.size(); ++ i) {
      const char *name = cfg[i].first.c_str();
      const char *val  = cfg[i].second.c_str();
      if (!strcmp(name, "data")) {
        flag = 1; continue;
      }
      if (!strcmp(name, "eval")) {
        evname = std::string(val);
        flag = 2; continue;
      }
      if (!strcmp(name, "pred")) {
        flag = 3; name_pred = val; continue;
      }
      if (!strcmp(name, "iter") && !strcmp(val, "end")) {
        utils::Assert(flag != 0, "wrong configuration file");
        if (flag == 1 && task != "pred") {
          utils::Assert(itr_train == NULL, "can only have one data");
          itr_train = cxxnet::CreateIterator(itcfg);
        }
        if (flag == 2 && task != "pred") {
          itr_evals.push_back(cxxnet::CreateIterator(itcfg));
          eval_names.push_back(evname);
        }
        if (flag == 3 && (task == "pred" || task == "pred_raw" ||
                          task == "extract")) {
          utils::Assert(itr_pred == NULL, "can only have one data:test");
          itr_pred = cxxnet::CreateIterator(itcfg);
        }
        flag = 0; itcfg.clear();
      }
      if (flag == 0) {
        defcfg.push_back(cfg[i]);
      }else{
        itcfg.push_back(cfg[i]);
      }
    }
    if (itr_train != NULL) {
      this->InitIter(itr_train, defcfg);
    }
    if (itr_pred != NULL) {
      this->InitIter(itr_pred, defcfg);
    }
    for (size_t i = 0; i < itr_evals.size(); ++ i) {
      this->InitIter(itr_evals[i], defcfg);
    }
  }
 private:
  inline void TaskPredict(void) {
    utils::Assert(itr_pred != NULL, "must specify a predict iterator to generate predictions");
    printf("start predicting...\n");
    FILE *fo = utils::FopenCheck(name_pred.c_str(), "w");
    itr_pred->BeforeFirst();
    mshadow::TensorContainer<mshadow::cpu, 1> pred;
    while (itr_pred->Next()) {
      const DataBatch& batch = itr_pred->Value();
      net_trainer->Predict(&pred, batch);
      utils::Assert(batch.num_batch_padd < batch.batch_size, "num batch pad must be smaller");
      mshadow::index_t sz = pred.size(0) - batch.num_batch_padd;
      for (mshadow::index_t j = 0; j < sz; ++j) {
        fprintf(fo, "%g\n", pred[j]);
      }
    }
    fclose(fo);
    printf("finished prediction, write into %s\n", name_pred.c_str());
  }
  inline void TaskExtractFeature() {
    long nrow = 0;
    mshadow::Shape<3> dshape;
    utils::Check(itr_pred != NULL,
                 "must specify a predict iterator to generate predictions");
    printf("start predicting...\n");
    FILE *fo = utils::FopenCheck(name_pred.c_str(), "wb");
    std::string name_meta = name_pred + ".meta";
    FILE *fm = utils::FopenCheck(name_meta.c_str(), "w");
    itr_pred->BeforeFirst();

    time_t start    = time(NULL);
    int sample_counter = 0;
    mshadow::TensorContainer<mshadow::cpu, 4> pred;
    while (itr_pred->Next()) {
      const DataBatch &batch = itr_pred->Value();
      if (extract_node_name != ""){
        net_trainer->ExtractFeature(&pred, batch, extract_node_name.c_str());
      } else {
        utils::Error("extract node name must be specified in task extract_feature.");
      }
      utils::Assert(batch.num_batch_padd < batch.batch_size, "num batch pad must be smaller");
      mshadow::index_t sz = pred.size(0) - batch.num_batch_padd;
      nrow += sz;
      for (mshadow::index_t j = 0; j < sz; ++j) {
        mshadow::Tensor<mshadow::cpu, 2> d = pred[j].FlatTo2D();
        for (mshadow::index_t k = 0; k < d.size(0); ++k) {
          if (output_format) {
            for (mshadow::index_t m = 0; m < d.size(1); ++m) {
              fprintf(fo, "%g ", d[k].dptr_[m]);
            }
          } else {
            fwrite(d[k].dptr_, sizeof(float), d.size(1), fo);
          }
        }
        if (output_format) {
          fprintf(fo, "\n");
        }
      }
      if (sz != 0) {
        dshape = pred[0].shape_;
      }
      if (++ sample_counter  % print_step == 0) {
        long elapsed = (long)(time(NULL) - start);
        if (!silent) {
          printf("\r                                                               \r");
          printf("batch:[%8d] %ld sec elapsed", sample_counter, elapsed);
          fflush(stdout);
        }
      }
    }
    long elapsed = (long)(time(NULL) - start);
    printf("\r                                                               \r");
    printf("batch:[%8d] %ld sec elapsed\n", sample_counter, elapsed);

    fclose(fo);
    fprintf(fm, "%ld,%u,%u,%u\n", nrow, dshape[0], dshape[1], dshape[2]);
    fclose(fm);
    printf("finished prediction, write into %s\n", name_pred.c_str());
  }
  inline void TaskTrain(void) {
    time_t start    = time(NULL);
    unsigned long elapsed = 0;
    if (continue_training == 0 && name_model_in == "NULL") {
      this->SaveModel();
    } else {
      if (!silent) {
        printf("continuing from round %d", start_counter-1);
        fflush(stdout);
      }
      for (size_t i = 0; i < itr_evals.size(); ++i) {
        std::string res = net_trainer->Evaluate(itr_evals[i], eval_names[i].c_str());
        fprintf(stderr, "%s", res.c_str());
      }
      fprintf(stderr, "\n");
      fflush(stderr);
    }
    
    if (itr_train != NULL) {
      if (test_io != 0) {
        printf("start I/O test\n");
      }
      int cc = max_round;
      while (start_counter <= num_round && cc --) {
        if (!silent) {
          printf("update round %d", start_counter -1); fflush(stdout);
        }
        int sample_counter = 0;
        net_trainer->StartRound(start_counter);
        itr_train->BeforeFirst();
        while (itr_train->Next()) {
          if (test_io == 0) {
            net_trainer->Update(itr_train->Value());
          }
          if (++ sample_counter  % print_step == 0) {
            elapsed = (long)(time(NULL) - start);
            if (!silent) {
              printf("\r                                                               \r");
              printf("round %8d:[%8d] %ld sec elapsed", start_counter-1,
                     sample_counter, elapsed);
              fflush(stdout);
            }
          }
        }

        if (test_io == 0) {
          // code handling evaluation
          fprintf(stderr, "[%d]", start_counter);
          // handle only with eval_train = 1, but not val data
          if (itr_evals.size() == 0) {
            std::string res = net_trainer->Evaluate(NULL, "train");
            fprintf(stderr, "%s", res.c_str());
          }
          for (size_t i = 0; i < itr_evals.size(); ++i) {
            std::string res = net_trainer->Evaluate(itr_evals[i], eval_names[i].c_str());
            fprintf(stderr, "%s", res.c_str());
          }
          fprintf(stderr, "\n");
          fflush(stderr);
        }
        elapsed = (unsigned long)(time(NULL) - start);
        this->SaveModel();
      }
      
      if (!silent) {
        printf("\nupdating end, %lu sec in all\n", elapsed);
      }
    }
  }

  inline void CopyModel(void){
    FILE *fi = utils::FopenCheck(name_model_in.c_str(), "rb");
    utils::Assert(fread(&net_type, sizeof(int), 1, fi) > 0, "loading model");
    net_trainer = this->CreateNet();
    utils::FileStream fs(fi);
    net_trainer->CopyModelFrom(fs);
    fclose(fi);
  }
 private:
  /*! \brief type of net implementation */
  int net_type;
  /*! \brief whether to force reset network implementation */
  int reset_net_type;
  /*! \brief trainer */
  nnet::INetTrainer *net_trainer;
  /*! \brief training iterator, prediction iterator */
  IIterator<DataBatch>* itr_train, *itr_pred;
  /*! \brief validation iterators */
  std::vector<IIterator<DataBatch>* > itr_evals;
  /*! \brief evaluation names */
  std::vector<std::string> eval_names;
 private:
  /*! \brief all the configurations */
  std::vector<std::pair<std::string, std::string> > cfg;
 private:
  /*! \brief whether test io only */
  int test_io;
  /*! \brief  how may samples before print information */
  int print_step;
  /*! \brief number of round to train */
  int num_round;
  /*! \brief maximum number of round to train */
  int max_round;
  /*! \brief continue from model folder */
  int continue_training;
  /*! \brief  whether to save model after each round */
  int save_period;
  /*! \brief  start counter of model */
  int start_counter;
  /*! \brief  whether to be silent */
  int silent;
  /*! \brief  device of the trainer */
  std::string device;
  /*! \brief  task of the job */
  std::string task;
  /*! \brief  input model name */
  std::string name_model_in;
  /*! \brief training data */
  std::string name_data;
  /*! \brief folder name of output */
  std::string name_model_dir;
  /*! \brief file name to write prediction */
  std::string name_pred;
  /*! \brief the layer name to be extracted */
  std::string extract_node_name;
  /*! \brief output format of network */
  int output_format;
 };
}  // namespace cxxnet

// general main for PS
int WorkerNodeMain(int argc, char *argv[]) {
  cxxnet::CXXNetLearnTask tsk;
  return tsk.Run(argc, argv);
}
