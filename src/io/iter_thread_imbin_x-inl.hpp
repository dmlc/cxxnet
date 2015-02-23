#ifndef ITER_THREAD_IMBIN_X_INL_HPP_
#define ITER_THREAD_IMBIN_X_INL_HPP_
/*!
 * \file cxxnet_iter_thread_imbin-inl.hpp
 * \brief threaded version of page iterator
 * \author Tianqi Chen
 */
#include "data.h"
#include <cstdlib>
#include "../utils/thread_buffer.h"
#include "../utils/utils.h"
#include "../utils/decoder.h"

namespace cxxnet {
/*! \brief thread buffer iterator */
class ThreadImagePageIteratorX: public IIterator<DataInst> {
public:
  ThreadImagePageIteratorX(void) {
    idx_ = 0;
    silent_ = 0;
    itr.SetParam("buffer_size", "2");
    img_conf_prefix_ = "";
    flag_ = true;
    label_width_ = 1;
    dist_num_worker_ = 0;
    dist_worker_rank_ = 0;
  }
  virtual ~ThreadImagePageIteratorX(void) {
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "image_shape")) {
      utils::Assert(sscanf(val, "%u,%u,%u", &shape_[1], &shape_[2], &shape_[3]) == 3,
                    "input_shape must be three consecutive integers without space example: 1,1,200 ");
    }
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
    if (!strcmp(name, "label_width")) {
      label_width_ = atoi(val);
    }
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
    itr.get_factory().data_shape = shape_;
    itr.get_factory().label_shape[1] = label_width_;
    itr.Init();
    this->BeforeFirst();
  }
  virtual void BeforeFirst(void) {
    itr.BeforeFirst();
    flag_ = true;
  }
  virtual bool Next(void) {
    if (!flag_) return flag_;
    flag_ = itr.Next(out_);
    return flag_;
  }
  virtual const DataInst &Value(void) const {
    return out_;
  }

protected:
  /*! \brief internal flag */
  bool flag_;
  /*! \brief internal index */
  int idx_;
  /*! \brief number of distributed worker */
  int dist_num_worker_, dist_worker_rank_;
  /*! \brief output data */
  DataInst out_;
  /*! \brief label-width */
  int label_width_;
  /*! \brief silent */
  int silent_;
  /*! \brief prefix path of image binary, path to input lst */
  // format: imageid label path
  std::vector<std::string> path_imgbin_, path_imglst_;
  /*! \brief configuration bing */
  std::string img_conf_prefix_, img_conf_ids_;
  /*! \brief raw image list */
  std::string raw_imglst_, raw_imgbin_;
  mshadow::Shape<4> shape_;
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
    // file stream for binary page
    utils::StdFile fi;
    // page ptr
    utils::BinaryPage* page_ptr;
    // list of bin path
    std::vector<std::string> path_imgbin;
    // list of img list path
    std::vector<std::string> path_imglst;
    // seq of list index
    std::vector<int> lst_idx;
    // seq of instance index
    std::vector<int> inst_idx;
    // jpeg decoders
    std::vector<utils::JpegDecoder> decoders;
    // buffer for decoders
    std::vector<mshadow::TensorContainer<cpu, 3, unsigned char> > decoder_buf;
    // buf of data instance inner id
    std::vector<unsigned> inst_id_buf;
    // data buffer
    mshadow::TensorContainer<cpu, 4, unsigned char> data_buf;
    // label buffer
    mshadow::TensorContainer<cpu, 2> label_buf;
    // image shape
    mshadow::Shape<4> data_shape;
    // label shape
    mshadow::Shape<2> label_shape;
    // id for list
    int idx;
    // id for data
    int didx;
    // end flag
    bool flag;
    // file ptr for list
    FILE *fp;
  public:
    Factory() : idx(0), didx(0), flag(true) {
      data_buf.set_pad(false);
      page_ptr = new utils::BinaryPage();
      // omp here
      decoders.resize(4);
      decoder_buf.resize(4);
    }
    inline bool Init() {
      lst_idx.resize(path_imgbin.size());
      for (unsigned int i = 0; i < path_imgbin.size(); ++i) {
        lst_idx[i] = i;
      }
      // shuffle lst_idx
      fi.Open(path_imgbin[lst_idx[idx]].c_str(), "rb");
      fp = utils::FopenCheck(path_imglst[lst_idx[idx]].c_str(), "r");
      didx = -1;
      return true;
    }
    inline void SetParam(const char *name, const char *val) {}

    inline bool FillBuf() {
      bool res = page_ptr->Load(fi);
      if (!res) return res;
      inst_idx.resize(page_ptr->Size());
      // unroll here
      for (int i = 0; i < page_ptr->Size(); ++i) {
        inst_idx[i] = i;
      }
      // shuffle inst_idx
      data_shape[0] = page_ptr->Size();
      label_shape[0] = page_ptr->Size();
      data_buf.Resize(data_shape);
      label_buf.Resize(label_shape);
      inst_id_buf.resize(page_ptr->Size());
      // omp here
      for (int i = 0; i < page_ptr->Size(); ++i) {
        utils::BinaryPage::Obj obj = (*page_ptr)[i];
        const int k = 0; // omp here
        decoders[k].Decode(static_cast<unsigned char*>(obj.dptr), obj.sz, &decoder_buf[k]);
        //swap channel, broadcast
        data_buf[i] = decoder_buf[k];
      }
      for (int i = 0; i < page_ptr->Size(); ++i) {
        utils::Assert(fscanf(fp, "%u", &inst_id_buf[i]) == 1, "invalid list format");
        for (unsigned int j = 0; j < label_shape[1]; ++j) {
          float tmp;
          utils::Check(fscanf(fp, "%f", &tmp) == 1,
                     "ImageList format:label_width=%u but only have %u labels per line",
                     label_shape[1], j);
          label_buf[i][j] = tmp;
        }
        utils::Assert(fscanf(fp, "%*[^\n]\n") == 0, "ignore");
      }
      didx = 0;
      return true;
    }

    inline bool LoadNext(DataInst &val) {
      if (!flag) return flag;
      if (didx == -1 || didx >= static_cast<int>(data_buf.size(0))) {
        bool res = this->FillBuf();
        if (res) {
          return this->LoadNext(val);
        } else {
          idx += 1;
          if (idx >= static_cast<int>(lst_idx.size())) {
            flag = false;
            return flag;
          } else {
            fi.Close();
            fi.Open(path_imgbin[lst_idx[idx]].c_str(), "rb");
            if (fp) fclose(fp);
            utils::FopenCheck(path_imglst[lst_idx[idx]].c_str(), "r");
            didx = -1;
            return this->LoadNext(val);
          }
        }
      } else {
        val.index = inst_idx[didx];
        val.data = mshadow::expr::tcast<float>(data_buf[didx]);
        mshadow::Copy(val.label, label_buf[didx++]);
        return true;
      }
    }
    inline DataInst Create(void) {
      DataInst a;
      a.data.shape_ = mshadow::Shape3(data_shape[1], data_shape[2], data_shape[3]);
      a.label.shape_ = mshadow::Shape1(label_shape[1]);
      mshadow::AllocSpace(&a.data, false);
      mshadow::AllocSpace(&a.label, false);
      return a;
    }
    inline void FreeSpace(DataInst &a) {
      mshadow::FreeSpace(&a.data);
      mshadow::FreeSpace(&a.label);
    }
    inline void Destroy() {
      fi.Close();
      if (fp) fclose(fp);
      delete page_ptr;
    }
    inline void BeforeFirst() {
      if (path_imgbin.size() == 1) {
        fi.Seek(0);
        if (fp) fclose(fp);
        fp = utils::FopenCheck(path_imglst[lst_idx[idx]].c_str(), "r");
      } else {
        // shuffle lst_idx
        fi.Close();
        fi.Open(path_imgbin[lst_idx[idx]].c_str(), "rb");
        if (fp) fclose(fp);
        fp = utils::FopenCheck(path_imglst[lst_idx[idx]].c_str(), "r");
      }
      flag = true;
      idx = 0;
      didx = -1;
    }
  };
protected:
  utils::ThreadBuffer<DataInst, Factory> itr;
}; // class ThreadImagePageIterator
}; // namespace cxxnet
#endif

