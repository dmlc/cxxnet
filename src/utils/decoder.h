#ifndef DECODER_H
#define DECODER_H
#pragma once

#include <vector>
#include <jpeglib.h>
#include <setjmp.h>
#include <jerror.h>
#include <mshadow/tensor.h>
#include "./utils.h"
#include "assert.h"
#if CXXNET_USE_OPENCV
  #include <opencv2/opencv.hpp>
#endif

namespace cxxnet {
namespace utils {

struct JpegDecoder {
public:
  JpegDecoder(void) {
    init = false;
  }

  ~JpegDecoder() {
    if (init) jpeg_destroy_decompress(&cinfo);
  }

  inline void Decode(unsigned char *ptr, size_t sz,
                     mshadow::TensorContainer<cpu, 3, unsigned char> *p_data) {
    if (!init) {
      init = true;
      cinfo.err = jpeg_std_error(&jerr.base);
      jerr.base.error_exit = jerror_exit;
      jerr.base.output_message = joutput_message;
      jpeg_create_decompress(&cinfo);
    }
    if(setjmp(jerr.jmp)) {
      jpeg_destroy_decompress(&cinfo);
      utils::Error("Libjpeg fail to decode");
    }
    this->jpeg_mem_src(&cinfo, ptr, sz);
    assert(jpeg_read_header(&cinfo, TRUE) == JPEG_HEADER_OK);
    assert(jpeg_start_decompress(&cinfo) == true);
    p_data->Resize(mshadow::Shape3(cinfo.output_width, cinfo.output_height, cinfo.output_components));
    JSAMPROW jptr = &((*p_data)[0][0][0]);
    while (cinfo.output_scanline < cinfo.output_height) {
      assert(jpeg_read_scanlines(&cinfo, &jptr, 1) == true);
      jptr += cinfo.output_width * cinfo.output_components;
    }
    assert(jpeg_finish_decompress(&cinfo) == true);
  }
private:
  struct jerror_mgr {
    jpeg_error_mgr base;
    jmp_buf jmp;
  };

  METHODDEF(void) jerror_exit(j_common_ptr jinfo) {
    jerror_mgr* err = (jerror_mgr*)jinfo->err;
    longjmp(err->jmp, 1);
  }

  METHODDEF(void) joutput_message(j_common_ptr) {}

  static boolean mem_fill_input_buffer (j_decompress_ptr cinfo) {
    #ifdef PROCESS_TRUNCATED_IMAGES
    jpeg_source_mgr* src = cinfo->src;
    static const JOCTET EOI_BUFFER[ 2 ] = { (JOCTET)0xFF, (JOCTET)JPEG_EOI };
    src->next_input_byte = EOI_BUFFER;
    src->bytes_in_buffer = sizeof( EOI_BUFFER );
    #else
    ERREXIT(cinfo, JERR_INPUT_EMPTY);
    #endif
    return true;
  }

  static void mem_skip_input_data (j_decompress_ptr cinfo, long num_bytes) {
    struct jpeg_source_mgr* src = (struct jpeg_source_mgr*) cinfo->src;
    if (num_bytes > 0) {
      src->next_input_byte += (size_t) num_bytes;
      src->bytes_in_buffer -= (size_t) num_bytes;
    }
    #ifdef PROCESS_TRUNCATED_IMAGES
    src->bytes_in_buffer = 0;
    #else
    ERREXIT( cinfo, JERR_INPUT_EOF );
    #endif
  }

  static void mem_term_source (j_decompress_ptr cinfo) {}
  static void mem_init_source (j_decompress_ptr cinfo) {}

  void jpeg_mem_src (j_decompress_ptr cinfo, void* buffer, long nbytes) {
    src.init_source = mem_init_source;
    src.fill_input_buffer = mem_fill_input_buffer;
    src.skip_input_data = mem_skip_input_data;
    src.resync_to_restart = jpeg_resync_to_restart;
    src.term_source = mem_term_source;
    src.bytes_in_buffer = nbytes;
    src.next_input_byte = (JOCTET*)buffer;
    cinfo->src = &src;
  }
private:
  struct jpeg_decompress_struct cinfo;
  jpeg_source_mgr src;
  jerror_mgr jerr;
  JSAMPARRAY buffer;
  bool init;
};

#if CXXNET_USE_OPENCV
struct OpenCVDecoder {
  void Decode(unsigned char *ptr, size_t sz, mshadow::TensorContainer<cpu, 3, unsigned char> *p_data) {
    buf.resize(sz);
    memcpy(&buf[0], ptr, sz);
    cv::Mat res = cv::imdecode(buf, 1);
    utils::Assert(res.data != NULL, "decoding fail");
    p_data->Resize(mshadow::Shape3(res.rows, res.cols, 3));
    for (int y = 0; y < res.rows; ++y) {
      for (int x = 0; x < res.cols; ++x) {
        cv::Vec3b bgr = res.at<cv::Vec3b>(y, x);
        // store in RGB order
        (*p_data)[y][x][2] = bgr[0];
        (*p_data)[y][x][1] = bgr[1];
        (*p_data)[y][x][0] = bgr[2];
      }
    }
    res.release();
  }
private:
  std::vector<unsigned char> buf;
};
#endif
} // namespace utils
} // namespace cxxnet

#endif // DECODER_H
