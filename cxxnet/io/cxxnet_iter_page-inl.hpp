#ifndef CXXNET_ITER_PAGE_INL_HPP
#define CXXNET_ITER_PAGE_INL_HPP
#pragma once

/*! \file cxxnet_iter_page-inl.hpp
 *  \brief implementation of page iterator
 *  \author Bing Xu
 */

#include "mshadow/tensor.h"
#include "cxxnet_data.h"
#include <opencv2/opencv.hpp>

namespace cxxnet {
    class PageIterator : public IIterator<DataInst> {
    public:
        PageIterator() {
            img_.set_pad(false);
            fplst_ = NULL;
            silent_ = 0;
            path_page_ = "";
            path_imglst_ = "";
            cnt_ = 0;
        }
        virtual ~PageIterator() {
            if (fplst_) fclose(fplst_);
        }
        virtual void SetParam( const char *name, const char *val ){
            if( !strcmp( name, "image_list" ) )    path_imglst_ = val;
            if( !strcmp( name, "image_bin") )      path_page_ = val;
            if( !strcmp( name, "silent"   ) )      silent_ = atoi( val );
        }
        virtual void Init() {
            fplst_ = utils::FopenCheck(path_imglst_.c_str(), "r");
            reader_.Open(path_page_.c_str(), "rb");
            if( silent_ == 0 ){
                printf("PageIterator:image_list=%s, bin=%s\n", path_imglst_.c_str(), path_page_.c_str() );
            }
            pg_.Load(reader_);
        }
        virtual void BeforeFirst() {
            fseek(fplst_, 0, SEEK_SET);
            reader_.Seek(0L);
            pg_.Load(reader_);
        }
        virtual bool Next() {
            while (fscanf(fplst_, "%u%f%*[^\n]\n", &out_.index, &out_.label) == 2) {
                while (cnt_ >= pg_.Size()) {
                    utils::Assert( pg_.Load( reader_), "input image_list containes more images than binary file" );
                }
                LoadImage(img_, out_, cnt_++);
                return true;
            }
            return false;
        }
        virtual const DataInst &Value() const {
            return out_;
        }
    protected:
        inline void LoadImage(mshadow::TensorContainer<cpu,3> &img,
                              DataInst &out,
                              int cnt) {
            utils::BinaryPage::Obj obj = pg_[cnt];
            tmp_data_.resize(obj.sz);
            std::copy((unsigned char*)obj.dptr, ((unsigned char*)obj.dptr) + obj.sz, tmp_data_.begin());
            cv::Mat res = cv::imdecode(cv::Mat(tmp_data_), 1);
            utils::Assert( res.data != NULL, "decoding fail" );

            img_.Resize(mshadow::Shape3(3, res.rows, res.cols));

            for( index_t y = 0; y < img.shape[1]; ++y ){
                for( index_t x = 0; x < img.shape[0]; ++x ){
                    cv::Vec3b bgr = res.at<cv::Vec3b>( y, x );
                    img[2][y][x] = bgr[0];
                    img[1][y][x] = bgr[1];
                    img[0][y][x] = bgr[2];
                }
            }
            out.data = img;
            res.release();
        }
    protected:
        DataInst out_;
        int silent_;
        FILE *fplst_;
        std::string path_page_, path_imglst_;
        mshadow::TensorContainer<cpu,3> img_;
        std::vector<unsigned char> tmp_data_;
        cxxnet::utils::BinaryPage pg_;
        cxxnet::utils::StdFile reader_;
        int cnt_;
    }; // class PageIterator

}; // namespace cxxnet

#endif // CXXNET_ITER_PAGE_INL_HPP
