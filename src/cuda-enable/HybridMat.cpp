//
// Created by boss on 2020-12-18.
//

#include "cuda-enable/HybridMat.h"

namespace cuda_enable {
    const cv::cuda::GpuMat &HybridMat::getGpuMat() const {
        return gpu_mat;
    }
}


