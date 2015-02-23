#ifndef DECODER_H
#define DECODER_H
#pragma once

#include <vector>
#include <jpeglib.h>
#include <setjmp.h>
#include <jerror.h>
#include <mshadow/tensor.h>
#include "./utils.h"

namespace cxxnet {
namespace utils {

struct JpegDecoder {
public:
  JpegDecoder() {
    cinfo.err = jpeg_std_error(&jerr.base);
    jerr.base.error_exit = jerror_exit;
    jerr.base.output_message = joutput_message;
    jpeg_create_decompress(&cinfo);
  }
  ~JpegDecoder() {
    jpeg_destroy_decompress(&cinfo);
  }
  void Decode(unsigned char *src, size_t sz, mshadow::TensorContainer<mshadow::cpu, 3, unsigned char> *p_data) {
    p_data->set_pad(false);
    this->jpeg_mem_src(&cinfo, src, sz); // for old version libjpeg
    utils::Check(jpeg_read_header(&cinfo, TRUE) == JPEG_HEADER_OK, "invalid jpeg header");
    utils::Check(jpeg_start_decompress(&cinfo) == true, "libjpeg error");
    p_data->Resize(mshadow::Shape3(cinfo.output_components, cinfo.output_height, cinfo.output_width));
    _Decode(&((*p_data)[0][0][0]));
  }
private:
  struct jerror_mgr {
    jpeg_error_mgr base;
    jmp_buf jmp;
  }; // struct jerror_mgr
private:
  struct jpeg_decompress_struct cinfo;
  jerror_mgr jerr;
private:
  void _Decode(JSAMPROW jptr) {
    while (cinfo.output_scanline < cinfo.output_height) {
      utils::Check(jpeg_read_scanlines(&cinfo, &jptr, 1) == true, "jpeg decoder failed");
      jptr += cinfo.output_width * cinfo.output_components;
    }
    utils::Check(jpeg_finish_decompress(&cinfo) == true, "jpeg decoder failed");
  }
  METHODDEF(void) jerror_exit(j_common_ptr jinfo) {
    jerror_mgr* err = (jerror_mgr*)jinfo->err;
    longjmp(err->jmp, 1);
  }
  METHODDEF(void) joutput_message(j_common_ptr) {}
  static void init_source (j_decompress_ptr cinfo) {}
  static boolean fill_input_buffer (j_decompress_ptr cinfo) {
    ERREXIT(cinfo, JERR_INPUT_EMPTY);
    return true;
  }
  static void skip_input_data (j_decompress_ptr cinfo, long num_bytes) {
    struct jpeg_source_mgr* src = (struct jpeg_source_mgr*) cinfo->src;
    if (num_bytes > 0) {
      src->next_input_byte += (size_t) num_bytes;
      src->bytes_in_buffer -= (size_t) num_bytes;
    }
  }
  static void term_source (j_decompress_ptr cinfo) {}
  static void jpeg_mem_src (j_decompress_ptr cinfo, void* buffer, long nbytes) {
    struct jpeg_source_mgr* src;
    if (cinfo->src == NULL) {
      cinfo->src = (struct jpeg_source_mgr *)
         (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
                                     sizeof(struct jpeg_source_mgr));
    }
    src = (struct jpeg_source_mgr*) cinfo->src;
    src->init_source = init_source;
    src->fill_input_buffer = fill_input_buffer;
    src->skip_input_data = skip_input_data;
    src->resync_to_restart = jpeg_resync_to_restart;
    src->term_source = term_source;
    src->bytes_in_buffer = nbytes;
    src->next_input_byte = (JOCTET*)buffer;
  }
}; // struct JpegDecoder

} // namespace utils
} // namespace cxxnet

#endif // DECODER_H
