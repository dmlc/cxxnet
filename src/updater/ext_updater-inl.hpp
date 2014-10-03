#ifndef CXXNET_UPDATER_EXT_UPDATER_INL_HPP_
#define CXXNET_UPDATER_EXT_UPDATER_INL_HPP_
/*!
 * \file ext_updater-inl.hpp
 * \brief implementation of some of experimental updater, usually not needed
 * \author Tianqi Chen
 */
#include <limits>
#include <mshadow/tensor.h>
#include "./updater.h"
#include "./param.h"
#include "../utils/global_random.h"

namespace cxxnet {
namespace updater {
template<typename xpu, int dim>
class NoiseSGDUpdater : public SGDUpdater<xpu, dim> {
 public:
  NoiseSGDUpdater(mshadow::Tensor<xpu,dim> w, mshadow::Tensor<xpu,dim> dw, 
                  const char *tag, mshadow::Random<xpu> *p_rnd)
      : Parent(w, dw, tag), prnd_(p_rnd), 
        noise_type_(1), sigma_(0.01f) {}
  virtual ~NoiseSGDUpdater() {}
  virtual void Init(void) {
    Parent::Init();
    if (Parent::param.silent == 0) {
      printf("NoiseSGDUpdater: eta=%f, mom=%f, sigma=%f\n", Parent::param.base_lr_, Parent::param.momentum, sigma_);
    }
  }
  virtual void Update(long epoch) {
    mshadow::Tensor<xpu,dim> &dw = Parent::dw;
    // multiplicative noise 
    switch (noise_type_) {
      case 1:{
        dw *= (1.0f + sigma_ * (prnd_->uniform(dw.shape) - 0.5f)); break;
      }
      case 0:{
        dw *= (1.0f + sigma_ * prnd_->gaussian(dw.shape)); break;
      }
      default: utils::Error("unknown noise type");
    }
    Parent::Update(epoch);
  }
  virtual void SetParam(const char *name, const char *val) {
    Parent::SetParam(name, val);
    if (!strcmp(name, "updater_noise") && !strcmp(val, "gaussian")) noise_type_ = 0;
    if (!strcmp(name, "updater_noise") && !strcmp(val, "uniform"))  noise_type_ = 1;
    if (!strcmp(name, "noise_sigma")) sigma_ = (float)atof(val);
  }
 private:
  typedef SGDUpdater<xpu, dim> Parent;
  mshadow::Random<xpu> *prnd_;
  // standard deviation
  int noise_type_;
  float sigma_;
};  // class NoiseSGDUpdater

template<typename xpu, int dim>
class SGHMCUpdater : public IUpdater<xpu> {
 public:
  SGHMCUpdater(mshadow::Random<xpu> *p_rnd, mshadow::Tensor<xpu,dim> w, mshadow::Tensor<xpu,dim> dw, const char *tag)
      : prnd(p_rnd), w(w), dw(dw) {
    param.tag = tag;
    m_w.Resize(w.shape, 0.0f);
    temp.Resize(w.shape);
  }
  virtual ~SGHMCUpdater(void) {}
  virtual void StartRound(int round) {
    param.round = round;
    param.hyper_sampled = 0;
  }
  virtual void Init(void) {
    if(param.silent == 0) {
      printf("SGDHMCUpdater: eta=%f, mom=%f\n", param.base_lr_, param.momentum);
    }
  }
  // update model parameters
  virtual void Update(long epoch) {
    param.ScheduleEpoch(epoch);
    if(param.need_hypersample() && param.hyper_sampled  == 0) {
      this->UpdateHyper(); param.hyper_sampled = 1;
    }
    m_w *= param.momentum;
    m_w += (-param.learning_rate) * (dw + param.wd * w);
    if(param.need_sample()) {
      m_w += prnd->gaussian(w.shape) * param.GetSigma();
    }
    w += m_w;
    // dw accumulate gradient instead of storing them, updater need to reset then to 0 after each update
    dw = 0.0f;
  }
  virtual void ApplyVisitor(typename IUpdater<xpu>::IVisitor *pvisitor) {
    pvisitor->Visit(param.tag.c_str(), w, dw);
  }
  // update hyper parameters
  virtual void UpdateHyper(void) {
    mshadow::Copy(temp, w);
    mshadow::Tensor<cpu,2> wt = temp.FlatTo2D();
    double sumcnt = wt.shape[1] * wt.shape[0];
    double sumsqr = 0.0f;
    // note: make the norm sum operation in mshadow
    for(index_t y = 0; y < wt.shape[1]; ++ y)
      for(index_t x = 0; x < wt.shape[0]; ++ x) {
        sumsqr += wt[y][x] * wt[y][x];
      }
    double alpha = param.hyper_alpha + 0.5 * sumcnt;
    double beta  = param.hyper_beta  + 0.5 * sumsqr;
    double plambda;
    if(param.temp < 1e-6f) {
      plambda = std::max(alpha - 1.0, 0.0) / beta;
    }else{
      plambda = utils::SampleGamma(alpha, beta);
    }
    // set weight decay
    param.wd = static_cast<float>(plambda / param.num_train);
    if(param.silent == 0 && param.print_hupdate != 0) {
      printf("hyperupdate[");
      for(int i = dim-1; i > 0 ; --i) {
        printf("%u,", temp.shape[i]);
      }
      printf("%u]:plambda=%f,wd=%f\n", temp.shape[0], plambda, param.wd);
    }
  }
  virtual void SetParam(const char *name, const char *val) {
    param.SetParam(name, val);
  }
 protected:
  struct HMCParam : public UpdaterParam {
    // when to start sample
    int start_sample;
    // when to start hyper parameter sampling
    int start_hsample;
    // number of training data
    int num_train;
    // Gamma(alpha, beta) prior on regularizer
    float hyper_alpha;
    float hyper_beta;
    // sample hyper parameter each gap_hsample over training data
    int gap_hsample;
    int hyper_sampled;
    // print hyper update
    int print_hupdate;
    // temperature
    float temp;
    // output precision matrix
    float lambda_output;
    // output preiction matrix
    HMCParam(void) {
      start_sample  = std::numeric_limits<int>::max();
      start_hsample = std::numeric_limits<int>::max();
      temp  = 1.0f;
      hyper_alpha = hyper_beta = 1.0f;
      gap_hsample = 1;
      lambda_output = 1.0f;
      hyper_sampled = 0;
      print_hupdate  = 0;
    }
    inline void SetParam(const char *name, const char* val) {
      UpdaterParam::SetParam(name, val);
      if(!strncmp(name, tag.c_str(), tag.length())) {
        if(name[tag.length()] == ':') name += tag.length() + 1;
      }
      if(!strcmp("start_sample", name))  start_sample = atoi(val);
      if(!strcmp("start_hsample", name)) start_hsample = atoi(val);
      if(!strcmp("gap_hsample", name))   gap_hsample = atoi(val);
      if(!strcmp("num_train", name))     num_train = atoi(val);
      if(!strcmp("temp", name))          temp = (float)atof(val);
      if(!strcmp("print_hupdate", name)) print_hupdate = atoi(val);
      if(!strcmp("lambda_output", name)) lambda_output = (float)atof(val);
    }
    inline bool need_sample(void) const {
      return round >= start_sample;
    }
    inline bool need_hypersample(void) const {
      int diff = round - start_hsample;
      return diff >= 0 && diff % gap_hsample == 0;
    }
    inline real_t GetSigma(void) const {
      real_t scale;
      if (momentum < 1e-6f) {
        scale = learning_rate / (num_train * lambda_output);
      } else {
        scale = learning_rate * (1.0f-momentum) / (num_train * lambda_output);
      }
      return std::sqrt(2.0f * temp * scale);
    }
  };

 private:
  // training parameter
  HMCParam param;
  // momentum variable
  mshadow::TensorContainer<xpu,dim> m_w;
  mshadow::TensorContainer<cpu,dim> temp;
  // PRNG
  mshadow::Random<xpu> *prnd;
  mshadow::Tensor<xpu,dim> w, dw;
};
}  // namespace updater
}  // namespace cxxnet
#endif

