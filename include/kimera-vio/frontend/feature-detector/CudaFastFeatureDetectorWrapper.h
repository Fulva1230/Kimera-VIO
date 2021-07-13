//
// Created by boss on 2020-12-17.
//

#ifndef KIMERA_VIO_CUDAFASTFEATUREDETECTORWRAPPER_H
#define KIMERA_VIO_CUDAFASTFEATUREDETECTORWRAPPER_H

#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>

class CudaFastFeatureDetectorWrapper : public cv::Feature2D{
public:
    explicit CudaFastFeatureDetectorWrapper(int threshold=10, bool nonmaxSuppression=true, int type=cv::FastFeatureDetector::TYPE_9_16, int max_npoints=5000);
    void detect(cv::InputArray image, std::vector<cv::KeyPoint> &keypoints, cv::InputArray mask = cv::noArray()) override;

private:
    cv::Ptr<cv::cuda::FastFeatureDetector> internal_detector;
};


#endif //KIMERA_VIO_CUDAFASTFEATUREDETECTORWRAPPER_H
