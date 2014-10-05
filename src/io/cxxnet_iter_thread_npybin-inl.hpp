#ifndef CXXNET_ITER_THREAD_NPYBIN_INL_HPP
#define CXXNET_ITER_THREAD_NPYBIN_INL_HPP
#pragma once
/*!
 * \file cxxnet_iter_thread_npybin-inl.hpp
 * \brief threaded version of page iterator
 * \author Bing Xu, Tianqi Chen
 */
#include <string>
#include "cxxnet_data.h"
#include "../utils/cxxnet_thread_buffer.h"
#include "../utils/cxxnet_io_utils.h"

namespace cxxnet {
/*! \brief thread buffer iterator */
class ThreadNpyPageIterator: public IIterator< DataInst > {
public:
  ThreadNpyPageIterator(void) {
    npy_.set_pad(false);
    fplst_ = NULL;
    silent_ = 0;
    path_npylst_ = "npy.lst";
    path_npybin_ = "npy.bin";
    itr.SetParam("buffer_size", "4");
    page_.page = NULL;
  }
  virtual ~ThreadNpyPageIterator(void) {
    if (fplst_ != NULL) fclose(fplst_);
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "npy_list"))    path_npylst_ = val;
    if (!strcmp(name, "npy_bin"))     path_npybin_ = val;
    if (!strcmp(name, "silent"))      silent_ = atoi(val);
  }
  virtual void Init(void) {
    fplst_  = utils::FopenCheck(path_npylst_.c_str(), "r");
    if (silent_ == 0) {
      printf("ThreadNpyPageIterator:npy_list=%s, bin=%s\n", path_npylst_.c_str(), path_npybin_.c_str());
    }
    itr.get_factory().fi.Open(path_npybin_.c_str(), "rb");
    itr.Init();
    this->BeforeFirst();
  }
  virtual void BeforeFirst(void) {
    fseek(fplst_ , 0, SEEK_SET);
    itr.BeforeFirst();
    this->LoadNextPage();
  }
  virtual bool Next(void) {
    while (fscanf(fplst_, "%u%f%*[^\n]\n", &out_.index, &out_.label) == 2) {
      this->NextBuffer(buf_);
      this->LoadNpy(npy_, out_, buf_);
      return true;
    }
    return false;
  }
  virtual const DataInst &Value(void) const {
    return out_;
  }
protected:
  inline static void LoadNpy(mshadow::TensorContainer<cpu, 3> &npy,
                             DataInst &out,
                             std::vector<unsigned char> &buf) {
    utils::Assert(buf.size() > 267, "Wrong Npy format");
    FILE *fp = fmemopen(&buf[0], buf.size(), "rb");
    char buffer[256];
    index_t shape[3];
    index_t loc1 = 0;
    index_t loc2 = 0;
    index_t ndims = 0;
    index_t sz = 1;
    index_t word_size = 0;
    fread(buffer, sizeof(char), 11, fp);
    std::string header = fgets(buffer, 256, fp);
    utils::Assert(header[header.size() - 1] == '\n');
    loc1 = header.find("(");
    loc2 = header.find(")");
    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    if (str_shape[str_shape.size() - 1] == ',') ndims = 1;
    else ndims = std::count(str_shape.begin(), str_shape.end(), ',') + 1;
    utils::Check(ndims == 3, "Unsupport dimension");
    for (index_t i = 0; i < ndims; ++i) {
      loc1 = str_shape.find(",");
      shape[i] = atoi(str_shape.substr(0, loc1).c_str());
      sz *= shape[i];
      str_shape = str_shape.substr(loc1 + 1);
    }
    loc1 = header.find("descr") + 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    utils::Assert(littleEndian, "Npy Endian error");
    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());
    utils::Assert(word_size == 4, "Npy file's dtype has to be np.float32");

    float *data = new float[sz];
    fread(data, word_size, sz, fp);
    npy.Resize(mshadow::Shape3(shape[2], shape[1], shape[0]));
    for (index_t x = 0; x < npy.shape[0]; ++x) {
      for (index_t y = 0; y < npy.shape[1]; ++y) {
        for (index_t c = 0; c < npy.shape[2]; ++c) {
          npy[c][y][x] = data[(x * shape[1] + y) * shape[2] + c];
        }
      }
    }
    out.data = npy;
    delete [] data;
    fclose(fp);
  }
  inline void NextBuffer(std::vector<unsigned char> &buf) {
    while (ptop_ >= page_.page->Size()) {
      this->LoadNextPage();
    }
    utils::BinaryPage::Obj obj = (*page_.page)[ ptop_ ];
    buf.resize(obj.sz);
    memcpy(&buf[0], obj.dptr, obj.sz);
    ++ ptop_;
  }
  inline void LoadNextPage(void) {
    utils::Assert(itr.Next(page_), "can not get first page");
    ptop_ = 0;
  }
protected:
  // output data
  DataInst out_;
  // silent
  int silent_;
  // file pointer to list file, information file
  FILE *fplst_;
  // prefix path of image binary, path to input lst, format: imageid label path
  std::string path_npybin_, path_npylst_;
  // temp storage for npy
  mshadow::TensorContainer<cpu, 3> npy_;
  // temp memory buffer
  std::vector<unsigned char> buf_;
private:
  struct PagePtr {
    utils::BinaryPage *page;
  };
  struct Factory {
  public:
    utils::StdFile fi;
  public:
    Factory() {}
    inline bool Init() {
      return true;
    }
    inline void SetParam(const char *name, const char *val) {}
    inline bool LoadNext(PagePtr &val) {
      return val.page->Load(fi);
    }
    inline PagePtr Create(void) {
      PagePtr a; a.page = new utils::BinaryPage();
      return a;
    }
    inline void FreeSpace(PagePtr &a) {
      delete a.page;
    }
    inline void Destroy() {
    }
    inline void BeforeFirst() {
      fi.Seek(0);
    }
  };
protected:
  PagePtr page_;
  int     ptop_;
  utils::ThreadBuffer<PagePtr, Factory> itr;
}; // class ThreadNpyPageIterator
}; // namespace cxxnet
#endif

