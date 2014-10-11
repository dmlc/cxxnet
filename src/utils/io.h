#ifndef CXXNET_UTILS_IO_H_
#define CXXNET_UTILS_IO_H_
/*!
 * \file io.h
 * \brief definition of abstract stream interface for IO
 * \author Bing Xu Tianqi Chen
 */
#include "./utils.h"
#include <zlib.h>
#include <string>
#include <algorithm>
#include <cstring>

namespace cxxnet {
namespace utils {
/*!
 * \brief interface of stream I/O, used to serialize model
 */
class IStream {
 public:
  /*!
   * \brief read data from stream
   * \param ptr pointer to memory buffer
   * \param size size of block
   * \return usually is the size of data readed
   */
  virtual size_t Read(void *ptr, size_t size) = 0;
  /*!
   * \brief write data to stream
   * \param ptr pointer to memory buffer
   * \param size size of block
   */
  virtual void Write(const void *ptr, size_t size) = 0;
  /*! \brief virtual destructor */
  virtual ~IStream(void) {}

 public:
  // helper functions to write various of data structures
  /*!
   * \brief binary serialize a vector 
   * \param vec vector to be serialized
   */
  template<typename T>
  inline void Write(const std::vector<T> &vec) {
    uint64_t sz = static_cast<uint64_t>(vec.size());
    this->Write(&sz, sizeof(sz));
    if (sz != 0) {
      this->Write(&vec[0], sizeof(T) * sz);
    }
  }
  /*!
   * \brief binary load a vector 
   * \param out_vec vector to be loaded
   * \return whether load is successfull
   */
  template<typename T>
  inline bool Read(std::vector<T> *out_vec) {
    uint64_t sz;
    if (this->Read(&sz, sizeof(sz)) == 0) return false;
    out_vec->resize(sz);
    if (sz != 0) {
      if (this->Read(&(*out_vec)[0], sizeof(T) * sz) == 0) return false;
    }
    return true;
  }
  /*!
   * \brief binary serialize a string
   * \param str the string to be serialized
   */ 
  inline void Write(const std::string &str) {
    uint64_t sz = static_cast<uint64_t>(str.length());
    this->Write(&sz, sizeof(sz));
    if (sz != 0) {
      this->Write(&str[0], sizeof(char) * sz);
    }
  }
  /*!
   * \brief binary load a string
   * \param out_str string to be loaded
   * \return whether load is successful
   */
  inline bool Read(std::string *out_str) {
    uint64_t sz;
    if (this->Read(&sz, sizeof(sz)) == 0) return false;
    out_str->resize(sz);
    if (sz != 0) {
      if (this->Read(&(*out_str)[0], sizeof(char) * sz) == 0) return false;
    }
    return true;
  }
  /*!
   * \brief read a simple type and return it
   *        for example fs.ReadType<int>() will read int from the stream
   * \return the data readed
   * \tparam TRet the type of data to be readed 
   */
  template<typename TRet>
  inline TRet ReadType(void) {
    TRet ret;
    this->Read(&ret, sizeof(ret));
    return ret;
  }
}; // class IStream

/*! \brief interface of i/o stream that support seek */
class ISeekStream: public IStream {
 public:
  /*! \brief seek to certain position of the file */
  virtual void Seek(size_t pos) = 0;
  /*! \brief tell the position of the stream */
  virtual size_t Tell(void) = 0;
};

/*! \brief a in memory buffer that can be read and write as stream interface */
struct MemoryBufferStream : public ISeekStream {
 public:
  MemoryBufferStream(std::string *p_buffer) 
      : p_buffer_(p_buffer) {
    curr_ptr_ = 0;
  }
  virtual ~MemoryBufferStream(void) {}
  virtual size_t Read(void *ptr, size_t size) {
    utils::Assert(curr_ptr_ <= p_buffer_->length(),
                  "read can not have position excceed buffer length");
    size_t nread = std::min(p_buffer_->length() - curr_ptr_, size);
    if (nread != 0) memcpy(ptr, &(*p_buffer_)[0] + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  virtual void Write(const void *ptr, size_t size) {
    if (size == 0) return;
    if (curr_ptr_ + size > p_buffer_->length()) {
      p_buffer_->resize(curr_ptr_+size);
    }
    memcpy(&(*p_buffer_)[0] + curr_ptr_, ptr, size); 
    curr_ptr_ += size;
  }
  virtual void Seek(size_t pos) {
    curr_ptr_ = static_cast<size_t>(pos);
  }
  virtual size_t Tell(void) {
    return curr_ptr_;
  }

 private:
  /*! \brief in memory buffer */
  std::string *p_buffer_;
  /*! \brief current pointer */
  size_t curr_ptr_;
}; // class MemoryBufferStream

struct GzFile : public ISeekStream {
 public:
  GzFile(const char *path, const char *mode) {
    fp_ = gzopen(path, mode);
    utils::Check(fp_ != NULL, "Failed to open file %s\n", path);
  }
  virtual ~GzFile(void) {
    this->Close();
  }
  virtual void Close(void) {
    if (fp_ != NULL){
      gzclose(fp_); fp_ = NULL;
    }
  }
  virtual size_t Read(void *ptr, size_t size) {
    return gzread(fp_, ptr, size);
  }
  virtual void Write(const void *ptr, size_t size) {
    gzwrite(fp_, ptr, size);
  }
  virtual void Seek(size_t pos) {
    gzseek(fp_, pos, SEEK_SET);
  }
  virtual size_t Tell(void) {
    return static_cast<size_t>(gztell(fp_));
  }
 private:
  gzFile fp_;
};

/*! \brief implementation of file i/o stream */
class FileStream : public ISeekStream {
 public:
  explicit FileStream(FILE *fp) : fp(fp) {}
  explicit FileStream(void) {
    this->fp = NULL;
  }
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, size, 1, fp);
  }
  virtual void Write(const void *ptr, size_t size) {
    std::fwrite(ptr, size, 1, fp);
  }
  virtual void Seek(size_t pos) {
    std::fseek(fp, pos, SEEK_SET);
  }
  virtual size_t Tell(void) {
    return std::ftell(fp);
  }
  inline void Close(void) {
    if (fp != NULL){
      std::fclose(fp); fp = NULL;
    }
  }

 protected:
  FILE *fp;
};

/*! \brief implementation of file i/o stream */
class StdFile: public ISeekStream {
 public:
  /*! \brief constructor */
  StdFile(const char *fname, const char *mode) {
    fp_ = utils::FopenCheck(fname, mode);
    fseek(fp_, 0L, SEEK_END);
    sz_ = ftell(fp_);
    fseek(fp_, 0L, SEEK_SET);
  }
  StdFile() {}
  virtual ~StdFile(void) {
    this->Close();
  }
  virtual size_t Read(void *ptr, size_t size) {
    return fread(ptr, size, 1, fp_);
  }
  virtual void Write(const void *ptr, size_t size) {
    fwrite(ptr, size, 1, fp_);
  }
  virtual void Seek(size_t pos) {
    fseek(fp_, pos, SEEK_SET);
  }
  virtual size_t Tell(void) {
    return static_cast<size_t>(ftell(fp_));
  }
  inline void Close(void) {
    if (fp_ != NULL){
      fclose(fp_); fp_ = NULL;
    }
  }
  inline size_t Size() {
    return sz_;
  }
 private:
  FILE *fp_;
  size_t sz_;
}; // class StdFile

/*! \brief Basic page class */
class BinaryPage {
 public:
  /*! \brief page size 64 MB */
  static const size_t kPageSize = 64 << 18;
 public:
  /*! \brief memory data object */
  struct Obj{
    /*! \brief pointer to the data*/
    void  *dptr;
    /*! \brief size */
    size_t sz;
    Obj(void * dptr, size_t sz) : dptr(dptr), sz(sz){}
  };
 public:
  /*! \brief constructor of page */
  BinaryPage(void)  {
    data_ = new int[kPageSize];
    utils::Check(data_ != NULL, "fail to allocate page, out of space");
    this->Clear();
  };
  ~BinaryPage() {
    if (data_) delete [] data_;
  }
  /*!
   * \brief load one page form instream
   * \return true if loading is successful
   */
  inline bool Load(utils::IStream &fi) {
    return fi.Read(&data_[0], sizeof(int)*kPageSize) !=0;
  }
  /*! \brief save one page into outstream */
  inline void Save(utils::IStream &fo) {
    fo.Write(&data_[0], sizeof(int)*kPageSize);
  }
  /*! \return number of elements */
  inline int Size(void){
    return data_[0];
  }
  /*! \brief Push one binary object into page
   *  \param fname file name of obj need to be pushed into
   *  \return false or true to push into
   */
  inline bool Push(const Obj &dat) {
    if(this->FreeBytes() < dat.sz + sizeof(int)) return false;
    data_[ Size() + 2 ] = data_[ Size() + 1 ] + dat.sz;
    memcpy(this->offset(data_[ Size() + 2 ]), dat.dptr, dat.sz);
    ++ data_[0];
    return true;
  }
  /*! \brief Clear the page */
  inline void Clear(void) {
    memset(&data_[0], 0, sizeof(int) * kPageSize);
  }
  /*!
   * \brief Get one binary object from page
   *  \param r r th obj in the page
   */
  inline Obj operator[](int r) {
    utils::Assert(r < Size(), "index excceed bound");
    return Obj(this->offset(data_[ r + 2 ]),  data_[ r + 2 ] - data_[ r + 1 ]);
  }
 private:
  /*! \return number of elements */
  inline size_t FreeBytes(void) {
    return (kPageSize - (Size() + 2)) * sizeof(int) - data_[ Size() + 1 ];
  }
  inline void* offset(int pos) {
    return (char*)(&data_[0]) + (kPageSize*sizeof(int) - pos);
  }
 private:
  //int data_[ kPageSize ];
  int *data_;
};  // class BinaryPage
}  // namespace utils
}  // namespace cxxnet
#endif
