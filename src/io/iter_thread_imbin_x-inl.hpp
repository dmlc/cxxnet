#ifndef ITER_THREAD_IMBIN_X_INL_HPP_
#define ITER_THREAD_IMBIN_X_INL_HPP_
/*!
 * \file cxxnet_iter_thread_imbin-inl.hpp
 * \brief threaded version of page iterator
 * \author Tianqi Chen
 */
#include "data.h"
#include <cstdlib>
#include <omp.h>
#include "../utils/thread_buffer.h"
#include "../utils/utils.h"
#include "../utils/decoder.h"
#include "../utils/global_random.h"

namespace cxxnet {
/*! \brief thread buffer iterator */
class ThreadImagePageIteratorX: public IIterator<DataInst> {
public:
  ThreadImagePageIteratorX(void) {
    silent_ = 0;
    itr.SetParam("buffer_size", "512");
    img_conf_prefix_ = "";
    dist_num_worker_ = 0;
    dist_worker_rank_ = 0;
  }
  virtual ~ThreadImagePageIteratorX(void) {
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
  struct Factory {
  public:
    // put everything in cache entry
    struct CacheEntry {
      // insance index
      unsigned inst_index;
      // label of each instance
      mshadow::TensorContainer<cpu, 1> label;
      // image data
      mshadow::TensorContainer<cpu, 3, unsigned char> img;
      CacheEntry() : label(false), img(false) {}
      CacheEntry(int label_width,
                 mshadow::Shape<3> imshape)
          : label(false), img(false) {
        label.Resize(mshadow::Shape1(label_width));
        img.Resize(imshape);
      }
    };
    // file stream for binary page
    utils::StdFile fi;
    // page ptr
    utils::BinaryPage page;
    // list of bin path
    std::vector<std::string> path_imgbin;
    // list of img list path
    std::vector<std::string> path_imglst;
    // seq of list index
    std::vector<int> list_order;
    // seq of inst index
    std::vector<int> inst_order;
    // jpeg decoders
    std::vector<utils::JpegDecoder*> decoders;
    // decoded data
    std::vector<CacheEntry> entry;
    // image shape
    mshadow::Shape<3> data_shape;
    /*! \brief label-width */
    int label_width_;
    // pointer for each list
    size_t list_ptr;
    // id for data
    int data_ptr;
    // number of decoding thread
    int nthread;
    // file ptr for list
    FILE *fplist;
    // shuffle
    int shuffle;
  public:
    Factory(void) {
      nthread = 3;
      label_width_ = 1;
      list_ptr = 0;
      data_ptr = 0;
      fplist = NULL;
      shuffle = 0;
      for (int i = 0; i < nthread; ++i) {
	decoders.push_back(new utils::JpegDecoder());
      }
    }
    ~Factory() {
      for (int  i = 0; i < nthread; ++i) {
	delete decoders[i];       
      }
    }
    inline bool Init(void) {
      list_order.resize(path_imgbin.size());
      for (size_t i = 0; i < path_imgbin.size(); ++i) {
        list_order[i] = i;
      }
      if (shuffle) {
        utils::Shuffle(list_order);
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
    }
    inline bool FillBuffer(void) {
      bool res = page.Load(fi);
      if (!res) return res;
      // always keep entry to maximum size to avoid re-allocation
      entry.resize(std::max(entry.size(),
                            static_cast<size_t>(page.Size())));
      if (shuffle) {
        inst_order.resize(std::max(entry.size(),
                                   static_cast<size_t>(page.Size())));
        for (size_t i = 0; i < inst_order.size(); ++i) {
          inst_order[i] = i;
        }
        utils::Shuffle(inst_order);
      }
      // omp here
      #pragma omp parallel for num_threads(nthread)
      for (int i = 0; i < page.Size(); ++i) {
        utils::BinaryPage::Obj obj = page[i];
        const int tid = omp_get_thread_num();
        decoders[tid]->Decode(static_cast<unsigned char*>(obj.dptr),
			      obj.sz,
			      &entry[i].img);
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
      a.data.shape_ = mshadow::Shape3(3, data_shape[1], data_shape[0]);
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
          val.index = entry[data_ptr].inst_index;
          mshadow::Copy(val.label, entry[data_ptr].label);
          utils::Check(val.data.shape_[2] == entry[data_ptr].img.shape_[0] &&
                       val.data.shape_[1] == entry[data_ptr].img.shape_[1],
                       "all the images in the bin must have same shape");
          if (entry[data_ptr].img.shape_[2] == 3) {
            val.data = tcast<real_t>(swapaxis<2, 0>(entry[data_ptr].img));
          } else {
            for (unsigned int i = 0; i < entry[data_ptr].img.size(1); ++i) {
              for (unsigned int j = 0; j < entry[data_ptr].img.size(0); ++j) {
                val.data[0][i][j] = static_cast<real_t>(entry[data_ptr].img[j][i][0]);
                val.data[1][i][j] = static_cast<real_t>(entry[data_ptr].img[j][i][0]);
                val.data[2][i][j] = static_cast<real_t>(entry[data_ptr].img[j][i][0]);
              }
            }
          }
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
        if (shuffle) {
          utils::Shuffle(list_order);
        }
        fi.Close();
        fi.Open(path_imgbin[list_order[0]].c_str(), "rb");
        if (fplist != NULL) fclose(fplist);
        fplist = utils::FopenCheck(path_imglst[list_order[0]].c_str(), "r");
      }
      utils::Check(this->FillBuffer(), "the first bin was empty");
    }
  };

protected:
  utils::ThreadBuffer<DataInst, Factory> itr;
}; // class ThreadImagePageIterator
}; // namespace cxxnet
#endif
