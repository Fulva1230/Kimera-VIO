//
// Created by boss on 2020-12-18.
//

#ifndef KIMERA_VIO_HYBRIDMAT_H
#define KIMERA_VIO_HYBRIDMAT_H

#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>

namespace cuda_enable {
    class HybridMat : public cv::Mat {
    public:
        template<class ...T>
        HybridMat(T &&... args):cv::Mat(std::forward<T>(args)...) {
            gpu_mat = cv::cuda::GpuMat(*this);
        }

        const cv::cuda::GpuMat &getGpuMat() const;

    private:
        cv::cuda::GpuMat gpu_mat;
    };
}


#endif //KIMERA_VIO_HYBRIDMAT_H
