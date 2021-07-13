//
// Created by boss on 2020-12-17.
//

#include "kimera-vio/frontend/feature-detector/CudaFastFeatureDetectorWrapper.h"

#include <opencv2/core/cuda.hpp>

CudaFastFeatureDetectorWrapper::CudaFastFeatureDetectorWrapper(int threshold, bool nonmaxSuppression, int type,
                                                               int max_npoints) {
    internal_detector = cv::cuda::FastFeatureDetector::create(threshold, nonmaxSuppression, type, max_npoints);
}

void CudaFastFeatureDetectorWrapper::detect(const cv::_InputArray &image, std::vector<cv::KeyPoint> &keypoints,
                                            const cv::_InputArray &mask) {
    cv::cuda::GpuMat gpu_image(image);
    cv::cuda::GpuMat gpu_mask(mask);
    internal_detector->detect(gpu_image, keypoints, gpu_mask);
}
