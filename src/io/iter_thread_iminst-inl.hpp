#ifndef ITER_THREAD_IMINST_INL_HPP
#define ITER_THREAD_IMINST_INL_HPP
#pragma once
#include "data.h"
#include <cstdlib>
#include "../utils/omp.h"
#include "./image_augmenter-inl.hpp"
#include "../utils/thread_buffer.h"
#include "../utils/utils.h"
#include "../utils/decoder.h"
#include "../utils/random.h"

namespace cxxnet {
/*! \brief thread buffer iterator */
class ThreadImageInstIterator: public IIterator<DataInst> {
public:
  ThreadImageInstIterator(void) {
    silent_ = 0;
    itr.SetParam("buffer_size", "4096");
    img_conf_prefix_ = "";
    dist_num_worker_ = 0;
    dist_worker_rank_ = 0;
  }
  virtual ~ThreadImageInstIterator(void) {
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "image_list")) {
      raw_imglst_ += val;
      raw_imglst_ += ",";
      path_imglst_.push_back(std::string(val));
    }
    if (!strcmp(name, "image_bin")) {
      raw_imgbin_ += val;
      raw_imgbin_ += ",";
      path_imgbin_.push_back(std::string(val));
    }
    if (!strcmp(name, "image_conf_prefix")) {
      img_conf_prefix_ = val;
    }
    if (!strcmp(name, "image_conf_ids")) {
      img_conf_ids_ = val;
    }
    if (!strcmp(name, "dist_num_worker")) {
      dist_num_worker_ = atoi(val);
    }
    if (!strcmp(name, "dist_worker_rank")) {
      dist_worker_rank_ = atoi(val);
    }
    if (!strcmp(name, "silent")) silent_ = atoi(val);
    itr.get_factory().SetParam(name, val);
  }
  virtual void Init(void) {
    this->ParseImageConf();
    if (silent_ == 0) {
      if (img_conf_prefix_.length() == 0) {
        printf("ThreadImagePageIterator:image_list=%s, bin=%s\n",
               raw_imglst_.c_str(), raw_imgbin_.c_str());
      } else {
        printf("ThreadImagePageIterator:image_conf=%s, image_ids=%s\n",
               img_conf_prefix_.c_str(), img_conf_ids_.c_str());
      }
    }
    utils::Check(path_imgbin_.size() == path_imglst_.size(),
                 "List/Bin number not consist");
    itr.get_factory().path_imgbin = path_imgbin_;
    itr.get_factory().path_imglst = path_imglst_;
    itr.Init();
    this->BeforeFirst();
  }
  virtual void BeforeFirst(void) {
    itr.BeforeFirst();
  }
  virtual bool Next(void) {
    return itr.Next(out_);
  }
  virtual const DataInst &Value(void) const {
    return out_;
  }

 protected:
  /*! \brief number of distributed worker */
  int dist_num_worker_, dist_worker_rank_;
  /*! \brief output data */
  DataInst out_;
  /*! \brief silent */
  int silent_;
  /*! \brief prefix path of image binary, path to input lst */
  // format: imageid label path
  std::vector<std::string> path_imgbin_, path_imglst_;
  /*! \brief configuration bing */
  std::string img_conf_prefix_, img_conf_ids_;
  /*! \brief raw image list */
  std::string raw_imglst_, raw_imgbin_;
  /*! \brief parse configure file */
  inline void ParseImageConf(void) {
    // handling for hadoop
    const char *ps_rank = getenv("PS_RANK");
    if (ps_rank != NULL) {
      this->SetParam("dist_worker_rank", ps_rank);
    }
    if (img_conf_prefix_.length() == 0) return;
    utils::Check(path_imglst_.size() == 0 &&
                 path_imgbin_.size() == 0,
                 "you can either set image_conf_prefix or image_bin/image_list");
    int lb, ub;
    utils::Check(sscanf(img_conf_ids_.c_str(), "%d-%d", &lb, &ub) == 2,
                 "image_conf_ids only support range, like 1-100");
    int n = ub + 1 - lb;
    if (dist_num_worker_ > 1) {
      int step = (n + dist_num_worker_ - 1) / dist_num_worker_;
      int begin = std::min(dist_worker_rank_ * step, n) + lb;
      int end = std::min((dist_worker_rank_ + 1) * step, n) + lb;
      lb = begin; ub = end - 1;
      utils::Check(lb <= ub,
                   "ThreadImagePageIterator: too many workers"\
                   "such that idlist cannot be divided between them");
    }
    for (int i = lb; i <= ub; ++i) {
      std::string tmp;
      tmp.resize(img_conf_prefix_.length() + 30);
      sprintf(&tmp[0], img_conf_prefix_.c_str(), i);
      tmp.resize(strlen(tmp.c_str()));
      path_imglst_.push_back(tmp + ".lst");
      path_imgbin_.push_back(tmp + ".bin");
    }
  }

private:
  struct InstEntry {
    // insance index
    unsigned inst_index;
    // label of each instance
    mshadow::TensorContainer<cpu, 1> label;
    // image data
    mshadow::TensorContainer<cpu, 3> img;
    // instance entry
    InstEntry() : label(false), img(false) {

    }
  };
  struct Factory {
  public:
    // put everything in cache entry
    struct CacheEntry {
      // insance index
      unsigned inst_index;
      // label of each instance
      mshadow::TensorContainer<cpu, 1> label;
      // image data
      mshadow::TensorContainer<cpu, 3> img;
      CacheEntry() : label(false), img(false) {}
      CacheEntry(int label_width,
                 mshadow::Shape<3> imshape)
          : label(false), img(false) {
        label.Resize(mshadow::Shape1(label_width));
        img.Resize(imshape);
      }
    };
    /*! \brief file stream for binary page */
    utils::StdFile fi;
    /*! \brief page ptr */
    utils::BinaryPage page;
    /*! \brief list of bin path */
    std::vector<std::string> path_imgbin;
    /*! \brief list of img list path */
    std::vector<std::string> path_imglst;
    /*! \brief seq of list index */
    std::vector<int> list_order;
    /*! \brief  seq of inst index */
    std::vector<int> inst_order;
    /*! \brief augmenters */
    std::vector<ImageAugmenter*> augmenters;
    /*! \brief random samplers */
    std::vector<utils::RandomSampler*> prnds;
    /*! \brief cache entry */
    std::vector<CacheEntry> entry;
    /*! \brief input image shape */
    mshadow::Shape<3> data_shape;
    /*! \brief label-width */
    int label_width_;
    /*! \brief pointer to current list */
    size_t list_ptr;
    /*! \brief  pointer to current data inst */
    int data_ptr;
    /*! \brief  number of decoding thread */
    int nthread;
    /*! \brief  file ptr for list file */
    FILE *fplist;
    /*! \brief shuffle flag */
    int shuffle;

  public:
    Factory(void) {
      nthread = 4;
      label_width_ = 1;
      list_ptr = 0;
      data_ptr = 0;
      fplist = NULL;
      shuffle = 0;
      rnd.Seed(kRandMagic);
      // setup decoders
      for (int i = 0; i < nthread; ++i) {
        augmenters.push_back(new ImageAugmenter());
        prnds.push_back(new utils::RandomSampler());
        prnds[i]->Seed((i + 1) * kRandMagic);
      }
    }
    ~Factory(void) {
      for (int  i = 0; i < nthread; ++i) {
        delete augmenters[i];
        delete prnds[i];
      }
    }
    inline bool Init(void) {
      list_order.resize(path_imgbin.size());
      for (size_t i = 0; i < path_imgbin.size(); ++i) {
        list_order[i] = i;
      }
      if (shuffle != 0) {
        rnd.Shuffle(list_order);
      }
      // load in data
      list_ptr = 0;
      fi.Open(path_imgbin[list_order[0]].c_str(), "rb");
      fplist = utils::FopenCheck(path_imglst[list_order[0]].c_str(), "r");
      utils::Check(this->FillBuffer(), "ImageIterator: first bin must be valid");
      // after init, we will know the data shape
      data_shape = entry[0].img.shape_;
      return true;
    }
    inline void SetParam(const char *name, const char *val) {
      if (!strcmp(name, "label_width")) {
        label_width_ = atoi(val);
      }
      if (!strcmp(name, "shuffle")) {
        shuffle = atoi(val);
      }
      if (!strcmp(name, "seed_data")) {
        rnd.Seed(atoi(val) + kRandMagic);
      }
      for (int i = 0; i < nthread; ++i) {
        augmenters[i]->SetParam(name, val);
      }
    }
    inline bool FillBuffer(void) {
      bool res = page.Load(fi);
      if (!res) return res;
      // always keep entry to maximum size to avoid re-allocation
      entry.resize(std::max(entry.size(),
                            static_cast<size_t>(page.Size())));
      inst_order.resize(page.Size());
      for (int i = 0; i < page.Size(); ++i) {
        inst_order[i] = i;
      }
      if (shuffle) {
        rnd.Shuffle(inst_order);
      }
      // omp here
      #pragma omp parallel for num_threads(nthread)
      for (int i = 0; i < page.Size(); ++i) {
        utils::BinaryPage::Obj obj = page[i];
        const int tid = omp_get_thread_num();
        augmenters[tid]->Process(static_cast<unsigned char*>(obj.dptr), obj.sz,
                                 &entry[i].img, prnds[tid]);
      }
      for (int i = 0; i < page.Size(); ++i) {
        utils::Check(fscanf(fplist, "%u", &entry[i].inst_index) == 1,
                      "invalid list format");
        entry[i].label.Resize(mshadow::Shape1(label_width_));
        for (int j = 0; j < label_width_; ++j) {
          float tmp;
          utils::Check(fscanf(fplist, "%f", &tmp) == 1,
                     "ImageList format:label_width=%u but only have %d labels per line",
                     label_width_, j);
          entry[i].label[j] = tmp;
        }
        utils::Assert(fscanf(fplist, "%*[^\n]\n") == 0, "ignore");
      }
      data_ptr = 0;
      return true;
    }
    inline DataInst Create(void) {
      DataInst a;
      a.data.shape_ = mshadow::Shape3(3, data_shape[1], data_shape[2]);
      a.label.shape_ = mshadow::Shape1(label_width_);
      mshadow::AllocSpace(&a.data, false);
      mshadow::AllocSpace(&a.label, false);
      return a;
    }
    inline void FreeSpace(DataInst &a) {
      mshadow::FreeSpace(&a.data);
      mshadow::FreeSpace(&a.label);
    }
    inline bool LoadNext(DataInst &val) {
      while (true) {
        if (data_ptr >= page.Size()) {
          if (!this->FillBuffer()) {
            list_ptr += 1;
            if (list_ptr >= list_order.size()) return false;
            fi.Close();
            fi.Open(path_imgbin[list_order[list_ptr]].c_str(), "rb");
            if (fplist != NULL) fclose(fplist);
            fplist = utils::FopenCheck(path_imglst[list_order[list_ptr]].c_str(), "r");
          }
        } else {
          using namespace mshadow::expr;
          val.index = entry[inst_order[data_ptr]].inst_index;
          mshadow::Copy(val.label, entry[inst_order[data_ptr]].label);
          mshadow::Copy(val.data, entry[inst_order[data_ptr]].img);
          data_ptr += 1;
          return true;
        }
      }
    }
    inline void Destroy() {
      fi.Close();
      if (fplist != NULL) fclose(fplist);
    }
    inline void BeforeFirst() {
      list_ptr = 0;
      if (path_imgbin.size() == 1) {
        fi.Seek(0);
        fseek(fplist, 0, SEEK_SET);
      } else {
        if (shuffle != 0) {
          rnd.Shuffle(list_order);
        }
        fi.Close();
        fi.Open(path_imgbin[list_order[0]].c_str(), "rb");
        if (fplist != NULL) fclose(fplist);
        fplist = utils::FopenCheck(path_imglst[list_order[0]].c_str(), "r");
      }
      utils::Check(this->FillBuffer(), "the first bin was empty");
    }

   private:
    utils::RandomSampler rnd;
    static const int kRandMagic = 111;
  };

protected:
  utils::ThreadBuffer<DataInst, Factory> itr;
}; // class ThreadImageInstIterator
}; // namespace cxxnet
#endif // ITER_THREAD_IMINST_INL_HPP
