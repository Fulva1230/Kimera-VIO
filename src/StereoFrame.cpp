/* ----------------------------------------------------------------------------
 * Copyright 2017, Massachusetts Institute of Technology,
 * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Luca Carlone, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file   Frame.h
 * @brief  Class describing a pair of stereo images
 * @author Luca Carlone
 */

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/video/tracking.hpp>
#include "StereoFrame.h"
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace VIO;

/* --------------------------------------------------------------------------------------- */
void StereoFrame::sparseStereoMatching(const int verbosity) {

  if (verbosity>0) {
    cv::Mat leftImgWithKeypoints = UtilsOpenCV::DrawCircles(left_frame_.img_, left_frame_.keypoints_);
    showImagesSideBySide(leftImgWithKeypoints,right_frame_.img_,"unrectifiedLeftWithKeypoints_", verbosity);
  }

  // Rectify images
  getRectifiedImages();

  // Get rectified left keypoints
  StatusKeypointsCV left_keypoints_rectified = undistortRectifyPoints(left_frame_.keypoints_,left_frame_.cam_param_,left_undistRectCameraMatrix_);

  // TODO (actually this is compensated later on in the pipeline): This should be correct but largely hinders the performance of RANSAC compensate versors for rectification
  //  gtsam::Rot3 camLrect_R_camL = UtilsOpenCV::Cvmat2rot(left_frame_.cam_param_.R_rectify_);
  //  std::cout << "left_frame_.cam_param_.R_rectify_ << " << camLrect_R_camL.matrix().determinant() << std::endl;
  //  for(size_t i = 0; i < left_frame_.versors_.size(); i++){ // for each versor
  //    left_frame_.versors_.at(i) = camLrect_R_camL.rotate(gtsam::Unit3(left_frame_.versors_.at(i))).unitVector(); // rotate versor
  //    // R_rectify = camL_R_camLrect' = camL_R_camLrect => camLrect_versor = camLrect_R_camL * camL_versor
  //    double normV = left_frame_.versors_.at(i).norm();
  //    left_frame_.versors_.at(i) = left_frame_.versors_.at(i) / normV;
  //    // renormalize to reduce error propagation
  //  }

  // **** now we get corresponding keypoints in right image ***************************
  // one way to go is to compute a dense disparity map,
  // COMMENT: This is beyond what we need, moreover, for GTSAM we'll also need the right keypoints
  // getDisparityImage(left_img_rectified_, right_img_rectified_, verbosity);
  //////////////////////////////////////////////////////////////////////////
  // another way is to find for sparse correspondences in right image using optical flow
  // COMMENT: this does not work, probably the baseline is too large
  // getRightKeypointsLKunrectified();
  //////////////////////////////////////////////////////////////////////////
  // another way is to extract corners in right and try to match with left
  // COMMENT: unfortunately, from visual inspection it seems that we do not get the same matches
  // KeypointsCV right_keypoints = UtilsOpenCV::ExtractCorners(right_frame_.img_);
  // cv::Mat rightImgWithKeypoints = UtilsOpenCV::DrawCircles(right_frame_.img_, right_keypoints);
  // showImagesSideBySide(leftImgWithKeypoints,rightImgWithKeypoints,"unrectifiedWithKeypoints", verbosity);

  //////////////////////////////////////////////////////////////////////////
  // the way we use is to find for sparse correspondences in right image using patch correlation along
  // (horizontal epipolar lines)
  double fx = left_undistRectCameraMatrix_.fx();
  // std::cout << "nr of template matching calls: " << left_frame_.getNrValidKeypoints() << std::endl;
  StatusKeypointsCV right_keypoints_rectified =
      getRightKeypointsRectified(left_img_rectified_, right_img_rectified_, left_keypoints_rectified, fx, baseline());

  // compute the depth for each keypoints
  keypoints_depth_ = getDepthFromRectifiedMatches(left_keypoints_rectified,right_keypoints_rectified, fx, baseline());

  // display
  if (verbosity>0) {
    cv::Mat left_rectifiedWithKeypoints = UtilsOpenCV::DrawCircles(left_img_rectified_, left_keypoints_rectified);
    drawEpipolarLines(left_rectifiedWithKeypoints, right_img_rectified_, 20, verbosity);
    cv::Mat right_rectifiedWithKeypoints = UtilsOpenCV::DrawCircles(right_img_rectified_, right_keypoints_rectified, keypoints_depth_);
    showImagesSideBySide(left_rectifiedWithKeypoints,right_rectifiedWithKeypoints,"rectifiedWithKeypointsAndDepth_", verbosity);
  }

  // store point pixels and statuses: for visualization and to populate the statuses
  std::tie(right_frame_.keypoints_, right_keypoints_status_) = DistortUnrectifyPoints(
      right_keypoints_rectified, right_frame_.cam_param_.undistRect_map_x_, right_frame_.cam_param_.undistRect_map_y_);

  // sanity check
  if(keypoints_depth_.size() != left_frame_.versors_.size())
    throw std::runtime_error("sparseStereoMatching: keypoints_depth_ & versors_ sizes are wrong!");

  // get 3D points and populate structures
  keypoints_3d_.clear();              keypoints_3d_.reserve(right_keypoints_rectified.size());
  left_keypoints_rectified_.clear();  left_keypoints_rectified_.reserve(right_keypoints_rectified.size());
  right_keypoints_rectified_.clear(); right_keypoints_rectified_.reserve(right_keypoints_rectified.size());
  // IMPORTANT: keypoints_3d_ are expressed in the rectified left frame, so we have
  // to compensate for rectification. We do not do this for the versors to avoid adding
  // numerical errors (we are using very tight thresholds on 5-point RANSAC)
  gtsam::Rot3 camLrect_R_camL = UtilsOpenCV::Cvmat2rot(left_frame_.cam_param_.R_rectify_);
  for(size_t i=0; i< right_keypoints_rectified.size(); i++){
    left_keypoints_rectified_.push_back(left_keypoints_rectified.at(i).second);
    right_keypoints_rectified_.push_back(right_keypoints_rectified.at(i).second);
    if(right_keypoints_rectified[i].first == Kstatus::VALID){
      Vector3 versor = camLrect_R_camL.rotate( left_frame_.versors_[i] );
      if(versor(2) < 1e-3) { throw std::runtime_error("sparseStereoMatching: found point with nonpositive depth!"); }
      keypoints_3d_.push_back(versor * keypoints_depth_[i] / versor(2)); //keypoints_depth_ is not the norm of the vector, it is the z component
    }else{
      keypoints_3d_.push_back(Vector3::Zero());
    }
  }

  // visualize statistics on the performance of the sparse stereo matching
  displayKeypointStats(right_keypoints_rectified);

  // sanity check
  checkStereoFrame();
}
/* --------------------------------------------------------------------------------------- */
void StereoFrame::checkStereoFrame() const {
  int nrLeftKeypoints = left_frame_.keypoints_.size();
  double tol = 1e-4;
  if(left_frame_.scores_.size() != nrLeftKeypoints)
    throw std::runtime_error("checkStereoFrame: left_frame_.scores.size()");

  if(right_frame_.keypoints_.size() != nrLeftKeypoints)
    throw std::runtime_error("checkStereoFrame: right_frame_.keypoints_.size()");

  if(right_keypoints_status_.size() != nrLeftKeypoints)
    throw std::runtime_error("checkStereoFrame: right_keypoints_status_.size()");

  if(keypoints_depth_.size() != nrLeftKeypoints)
    throw std::runtime_error("checkStereoFrame: keypoints_depth_.size()");

  if(keypoints_3d_.size() != nrLeftKeypoints)
    throw std::runtime_error("checkStereoFrame: keypoints_3d_.size()");

  if(left_keypoints_rectified_.size() != nrLeftKeypoints)
    throw std::runtime_error("checkStereoFrame: left_keypoints_rectified_.size()");

  if(right_keypoints_rectified_.size() != nrLeftKeypoints)
    throw std::runtime_error("checkStereoFrame: right_keypoints_rectified_.size()");

  for(size_t i=0; i< nrLeftKeypoints; i++){
    if(right_keypoints_status_[i]==Kstatus::VALID &&
        fabs(right_keypoints_rectified_[i].y - left_keypoints_rectified_[i].y) > 3){
      std::cout << "checkStereoFrame: rectified keypoints have different y " <<right_keypoints_rectified_[i].y
          << " vs. " << left_keypoints_rectified_[i].y << std::endl;
      throw std::runtime_error("checkStereoFrame: rectified keypoints have different y");
    }

    if(fabs( keypoints_3d_[i](2) - keypoints_depth_[i] ) > tol){
      std::cout << "checkStereoFrame: keypoints_3d_[i] has wrong depth " << keypoints_3d_[i](2) << " vs. " << keypoints_depth_[i] << std::endl;
      throw std::runtime_error("checkStereoFrame: keypoints_3d_[i] has wrong depth");
    }
    if(right_keypoints_status_[i] == Kstatus::VALID){
      if(fabs(right_frame_.keypoints_[i].x) + fabs(right_frame_.keypoints_[i].y) == 0)
        throw std::runtime_error("checkStereoFrame: right_frame_.keypoints_[i] is zero");

      if(keypoints_depth_[i] <= 0){ // also: cannot have zero depth!
        std::cout << "checkStereoFrame: keypoints_3d_[i] has nonpositive for valid point: " << keypoints_depth_[i] << std::endl;
        std::cout << "right_keypoints_status_[i] " << right_keypoints_status_[i] << std::endl;
        std::cout << "left_frame_.keypoints_[i] " << left_frame_.keypoints_[i] << std::endl;
        std::cout << "right_frame_.keypoints_[i] " << right_frame_.keypoints_[i] << std::endl;
        std::cout << "left_keypoints_rectified_[i] " << left_keypoints_rectified_[i] << std::endl;
        std::cout << "right_keypoints_rectified_[i] " << right_keypoints_rectified_[i] << std::endl;
        std::cout << "keypoints_depth_[i] " << keypoints_depth_[i] << std::endl;
        throw std::runtime_error("checkStereoFrame: keypoints_3d_[i] has nonpositive for valid point");
      }
    }else{
      if(keypoints_depth_[i] > 0){
        std::cout << "checkStereoFrame: keypoints_3d_[i] has positive for nonvalid point: " << keypoints_depth_[i] << std::endl;
        throw std::runtime_error("checkStereoFrame: keypoints_3d_[i] has positive for nonvalid point");
      }
    }
  }
}
/* --------------------------------------------------------------------------------------- */
std::pair<KeypointsCV, std::vector<Kstatus>> StereoFrame::DistortUnrectifyPoints(
    const StatusKeypointsCV keypoints_rectified, const cv::Mat map_x, const cv::Mat map_y)
{
  std::vector<Kstatus> pointStatuses;
  KeypointsCV points;
  for(size_t i=0; i< keypoints_rectified.size(); i++){
    pointStatuses.push_back(keypoints_rectified[i].first);
    if(keypoints_rectified[i].first == Kstatus::VALID){
      KeypointCV px = keypoints_rectified[i].second;
      auto x = map_x.at<float>(round(px.y),round(px.x));
      auto y = map_y.at<float>(round(px.y),round(px.x));
      points.push_back( KeypointCV(x,y) );
    }else{
      points.push_back( KeypointCV(0.0,0.0) );
    }
  }
  return std::make_pair(points,pointStatuses);
}
/* --------------------------------------------------------------------------------------- */
StatusKeypointsCV StereoFrame::undistortRectifyPoints(KeypointsCV left_keypoints_unrectified, const CameraParams cam_param,
    const gtsam::Cal3_S2 rectCameraMatrix) const
{
  StatusKeypointsCV left_keypoints_rectified;
  int invalidCount = 0;
  left_keypoints_rectified.reserve(left_keypoints_unrectified.size());
  for (size_t i = 0; i < left_keypoints_unrectified.size(); ++i){ // for each keypoint
    KeypointCV px = left_keypoints_unrectified[i];

    // the following undistort to a versor, then we can project by the new camera matrix
    Vector3 calibrated_versor = Frame::CalibratePixel(px, cam_param);

    // compensate for rectification
    gtsam::Rot3 R_rect = UtilsOpenCV::Cvmat2rot(cam_param.R_rectify_);
    calibrated_versor = R_rect.matrix() * calibrated_versor;

    // normalize to unit z
    if (fabs(calibrated_versor(2)) > 1e-4){
      calibrated_versor = calibrated_versor / calibrated_versor(2);
    }else{
      throw std::runtime_error("undistortRectifyPoints: versor with zero depth");
    }

    double fx = rectCameraMatrix.fx();
    double cx = rectCameraMatrix.px();
    double fy = rectCameraMatrix.fy();
    double cy = rectCameraMatrix.py();

    // rectified_versor = rectCameraMatrix * calibrated_versor; -> this is down manually since
    // the matrix and the versor are incompatible types (opencv vs. gtsam)
    KeypointCV px_undistRect = KeypointCV(fx*calibrated_versor(0)+cx,fy*calibrated_versor(1)+cy);
    px_undistRect = UtilsOpenCV::CropToSize(px_undistRect,cam_param.undistRect_map_x_.size());

    // sanity check: you can go back to the original image accurately
    auto x_check = cam_param.undistRect_map_x_.at<float>(round(px_undistRect.y),round(px_undistRect.x));
    auto y_check = cam_param.undistRect_map_y_.at<float>(round(px_undistRect.y),round(px_undistRect.x));

    float tol = 2.0; // pixels
    if (fabs(px.x - x_check)>tol || fabs(px.y - y_check)>tol){
      //std::cout << "undistortRectifyPoints: pixel mismatch" << std::endl;
      //std::cout << "px.x " << px.x << " x_check " << x_check << std::endl;
      //std::cout << "px.y " << px.y << " y_check " << y_check << std::endl;
      //std::cout << "px_undistRect " <<px_undistRect  << std::endl;
      //std::cout << "rounded " <<round(px_undistRect.y) << " " << round(px_undistRect.x)  << std::endl;
      //std::cout << "cam_param.undistRect_map_x_ " << cam_param.undistRect_map_x_.size() << std::endl;
      //std::cout << "map type " << cam_param.undistRect_map_x_.type() << std::endl;
      //std::cout << "fx " << fx << " cx " << cx << " fy " << fy << " cy " << cy << std::endl;
      //std::cout << "calibrated_versor\n" << calibrated_versor << std::endl;
      invalidCount += 1;
      left_keypoints_rectified.push_back(std::make_pair(Kstatus::NO_LEFT_RECT,px_undistRect)); // invalid points
    }else{ // point is valid!
      left_keypoints_rectified.push_back(std::make_pair(Kstatus::VALID,px_undistRect)); // valid points
    }
  }
#ifdef STEREO_FRAME_DEBUG_COUT
  if(invalidCount>0)
    std::cout << "undistortRectifyPoints: unable to match " << invalidCount << " keypoints" << std::endl;
#endif

  return left_keypoints_rectified;
}

/* --------------------------------------------------------------------------------------- */
cv::Mat StereoFrame::getDisparityImage(const cv::Mat imgLeft, const cv::Mat imgRight, const int verbosity) const
{
    throw std::runtime_error("getDisparityImage: not implemented");
//  cv::Mat imgDisparity16S = cv::Mat(imgLeft.rows, imgLeft.cols, CV_16S);
//  cv::Mat imgDisparity8U = cv::Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
//
//  //-- Call the constructor for StereoBM
//  int ndisparities = 16 * 5;   /**< Range of disparity */
//  int SADWindowSize = 21; /**< Size of the block window. Must be odd */
//  cv::StereoBM sbm(cv::StereoBM::BASIC_PRESET,ndisparities,SADWindowSize);
//
//  //-- Calculate the disparity image
//  sbm(imgLeft, imgRight, imgDisparity16S, CV_16S);
//
//  //-- Check its extreme values
//  double minVal; double maxVal;
//  minMaxLoc(imgDisparity16S, &minVal, &maxVal);
//
//  //-- Display it as a CV_8UC1 image
//  imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));
//
//  cv::namedWindow("disparity", cv::WINDOW_NORMAL);
//  cv::imshow("disparity", imgDisparity8U);
//
//  //-- Save the image
//  if(verbosity>1){
//    std::string img_name = "./outputImages/disparity_" + std::to_string(id_) + ".png";
//    cv::imwrite(img_name, imgDisparity16S);
//  }
//  cv::waitKey(50);
//  return imgDisparity16S;
}

/* --------------------------------------------------------------------------------------- */
void StereoFrame::cloneRectificationParameters(const StereoFrame& sf) {
  left_frame_.cam_param_.R_rectify_ = sf.left_frame_.cam_param_.R_rectify_;
  right_frame_.cam_param_.R_rectify_ = sf.right_frame_.cam_param_.R_rectify_;
  B_Pose_camLrect = sf.B_Pose_camLrect;
  baseline_ = sf.baseline_;
  left_frame_.cam_param_.undistRect_map_x_ = sf.left_frame_.cam_param_.undistRect_map_x_.clone();
  left_frame_.cam_param_.undistRect_map_y_ = sf.left_frame_.cam_param_.undistRect_map_y_.clone();
  right_frame_.cam_param_.undistRect_map_x_ = sf.right_frame_.cam_param_.undistRect_map_x_.clone();
  right_frame_.cam_param_.undistRect_map_y_ = sf.right_frame_.cam_param_.undistRect_map_y_.clone();
  left_frame_.cam_param_.P_ = sf.left_frame_.cam_param_.P_.clone();
  right_frame_.cam_param_.P_ = sf.right_frame_.cam_param_.P_.clone();
  left_undistRectCameraMatrix_ = sf.left_undistRectCameraMatrix_;
  right_undistRectCameraMatrix_ = sf.right_undistRectCameraMatrix_;
  isRectified_ = true;
#ifdef STEREO_FRAME_DEBUG_COUT
  std::cout << "cloned undistRect maps and other rectification parameters!" << std::endl;
#endif
}
/* --------------------------------------------------------------------------------------- */
void StereoFrame::computeRectificationParameters() { // note also computes the rectification maps

  // get extrinsics in open CV format
  cv::Mat L_Rot_R, L_Tran_R;
  // NOTE: openCV pose convention is the opposite, that's why we have to invert
  boost::tie(L_Rot_R,L_Tran_R) = UtilsOpenCV::Pose2cvmats(camL_Pose_camR.inverse()); // set L_Rot_R,L_Tran_R

  //////////////////////////////////////////////////////////////////////////
  // get rectification matrices
  CameraParams& left_camera_info = left_frame_.cam_param_;
  CameraParams& right_camera_info = right_frame_.cam_param_;

  cv::Mat P1,P2,Q; // P1 and P2 are the new camera matrices, but with an extra 0 0 0 column
  cv::stereoRectify(left_camera_info.camera_matrix_, left_camera_info.distortion_coeff_,
      right_camera_info.camera_matrix_, right_camera_info.distortion_coeff_,
      left_camera_info.image_size_, L_Rot_R, L_Tran_R,
      // following are output
      left_camera_info.R_rectify_,right_camera_info.R_rectify_,P1,P2,Q);

#ifdef STEREO_FRAME_DEBUG_COUT
  std::cout << "RESULTS OF RECTIFICATION: " << std::endl;
  std::cout << "left_camera_info.R_rectify_\n" << left_camera_info.R_rectify_ << std::endl;
  std::cout << "right_camera_info.R_rectify_\n" << right_camera_info.R_rectify_ << std::endl;
#endif

  // left camera pose after rectification
  gtsam::Rot3 camL_Rot_camLrect = UtilsOpenCV::Cvmat2rot(left_camera_info.R_rectify_).inverse();
  gtsam::Pose3 camL_Pose_camLrect = gtsam::Pose3(camL_Rot_camLrect,gtsam::Point3());
  B_Pose_camLrect = (left_camera_info.body_Pose_cam_).compose(camL_Pose_camLrect);

  // right camera pose after rectification
  gtsam::Rot3 camR_Rot_camRrect = UtilsOpenCV::Cvmat2rot(right_camera_info.R_rectify_).inverse();
  gtsam::Pose3 camR_Pose_camRrect = gtsam::Pose3(camR_Rot_camRrect,gtsam::Point3());
  gtsam::Pose3 B_Pose_camRrect = (right_camera_info.body_Pose_cam_).compose(camR_Pose_camRrect);

  // relative pose after rectification
  gtsam::Pose3 camLrect_Pose_calRrect = B_Pose_camLrect.between(B_Pose_camRrect);
  // get baseline
  baseline_ = camLrect_Pose_calRrect.translation().x();
  if(baseline_ > 1.1 * sparseStereoParams_.nominalBaseline || baseline_ < 0.9 * sparseStereoParams_.nominalBaseline){ // within 10% of the expected baseline
#ifdef STEREO_FRAME_DEBUG_COUT
    std::cout << "getRectifiedImages: abnormal baseline: " << baseline_  << ", nominalBaseline: " << sparseStereoParams_.nominalBaseline
        <<  "(+/-10%) \n\n\n\n" << std::endl;
#endif
    // throw std::runtime_error("getRectifiedImages: abnormal baseline");
  }

  // sanity check
  if(gtsam::Rot3::Logmap(camLrect_Pose_calRrect.rotation()).norm() > 1e-5 ){
    std::cout << "camL_Pose_camR log: " << gtsam::Rot3::Logmap(camL_Pose_camR.rotation()).norm() << std::endl;
    std::cout << "camLrect_Pose_calRrect log: " << gtsam::Rot3::Logmap(camLrect_Pose_calRrect.rotation()).norm() << std::endl;
    throw std::runtime_error("Vio constructor: camera poses do not seem to be rectified (rot)");
  }
  if(fabs(camLrect_Pose_calRrect.translation().y()) > 1e-3 || fabs(camLrect_Pose_calRrect.translation().z()) > 1e-3){
    camLrect_Pose_calRrect.print("camLrect_Pose_calRrect\n");
    throw std::runtime_error("Vio constructor: camera poses do not seem to be rectified (tran)");
  }

  //////////////////////////////////////////////////////////////////////////
  // get rectification & undistortion maps
  cv::initUndistortRectifyMap(left_camera_info.camera_matrix_,
      left_camera_info.distortion_coeff_, left_camera_info.R_rectify_, P1, left_camera_info.image_size_, CV_32FC1,
      // output:
      left_camera_info.undistRect_map_x_, left_camera_info.undistRect_map_y_);

  cv::initUndistortRectifyMap(right_camera_info.camera_matrix_,
      right_camera_info.distortion_coeff_, right_camera_info.R_rectify_, P2, right_camera_info.image_size_, CV_32FC1,
      // output:
      right_camera_info.undistRect_map_x_, right_camera_info.undistRect_map_y_);

  // store intermediate results from rectification
  left_camera_info.P_ = P1; // contains an extra column to project in homogeneous coordinates
  right_camera_info.P_ = P2; // contains an extra column to project in homogeneous coordinates
  left_undistRectCameraMatrix_ = UtilsOpenCV::Cvmat2Cal3_S2(P1); // this cuts the last column
  right_undistRectCameraMatrix_ = UtilsOpenCV::Cvmat2Cal3_S2(P2); // this cuts the last column
  isRectified_ = true;
#ifdef STEREO_FRAME_DEBUG_COUT
  std::cout << "storing undistRect maps and other rectification parameters!" << std::endl;
#endif

}

/* --------------------------------------------------------------------------------------- */
void StereoFrame::getRectifiedImages() {

  if(!isRectified_) // if we haven't computed rectification parameters yet
    computeRectificationParameters();

  if(left_frame_.img_.rows != left_img_rectified_.rows || left_frame_.img_.cols != left_img_rectified_.cols ||
      right_frame_.img_.rows != right_img_rectified_.rows || right_frame_.img_.cols != right_img_rectified_.cols ){ // if we haven't rectified images yet
    // rectify and undistort images
    cv::remap(left_frame_.img_, left_img_rectified_, left_frame_.cam_param_.undistRect_map_x_, left_frame_.cam_param_.undistRect_map_y_, cv::INTER_LINEAR);
    cv::remap(right_frame_.img_, right_img_rectified_, right_frame_.cam_param_.undistRect_map_x_, right_frame_.cam_param_.undistRect_map_y_, cv::INTER_LINEAR);
  }

  // rectification check:
  //  std::cout << "size before (left): " << left_frame_.img_.rows << " x " << left_frame_.img_.cols << std::endl;
  //  std::cout << "size after  (left): " << left_img_rectified_.rows << " x " << left_img_rectified_.cols << std::endl;
  //
  //  std::cout << "size before (right): " << right_frame_.img_.rows << " x " << right_frame_.img_.cols << std::endl;
  //  std::cout << "size after  (right): " << right_img_rectified_.rows << " x " << right_img_rectified_.cols << std::endl;
  //
  //  cv::namedWindow("left_frame_img_", cv::WINDOW_NORMAL);
  //  cv::imshow("left_frame_img_", left_frame_.img_);
  //
  //  cv::namedWindow("left_img_rectified_", cv::WINDOW_NORMAL);
  //  cv::imshow("left_img_rectified_", left_img_rectified_);
  //  cv::waitKey(0);
  //
  //  cv::namedWindow("right_frame_img_", cv::WINDOW_NORMAL);
  //  cv::imshow("right_frame_img_", right_frame_.img_);
  //
  //  cv::namedWindow("right_img_rectified_", cv::WINDOW_NORMAL);
  //  cv::imshow("right_img_rectified_", right_img_rectified_);
  //  cv::waitKey(0);
}
/* --------------------------------------------------------------------------------------- */
void StereoFrame::getRightKeypointsLKunrectified() {

  Frame& ref_frame = left_frame_;
  Frame& cur_frame = right_frame_;

  if(left_frame_.keypoints_.size() == 0)
    throw std::runtime_error("computeStereo: no keypoints found");

  // get correspondences on right image by using Lucas Kanade
  // Parameters
  int klt_max_iter = 40;
  double klt_eps = 0.001;
  int klt_win_size = 31;

  ////////////////////////////////////////////////////////////////////////
  // Setup termination criteria for optical flow
  std::vector<uchar> status;
  std::vector<float> error;
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
      klt_max_iter, klt_max_iter);

  // Fill up structure for reference pixels and their labels
  KeypointsCV px_ref;
  px_ref.reserve(ref_frame.keypoints_.size());
  for (size_t i = 0; i < ref_frame.keypoints_.size(); ++i)
    px_ref.push_back(ref_frame.keypoints_[i]);

  // Initialize to old locations
  KeypointsCV px_cur = px_ref;
  if(px_cur.size() > 0){
    // Do the actual tracking, so px_cur becomes the new pixel locations
    cv::calcOpticalFlowPyrLK(ref_frame.img_, cur_frame.img_,
        px_ref, px_cur, status, error,
        cv::Size2i(klt_win_size, klt_win_size),
        4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);
  }else{
    throw std::runtime_error("computeStereo: no available keypoints for stereo computation");
  }

  cur_frame.keypoints_.clear();
  int nrValidDepths = 0;
  for (int i = 0; i < px_ref.size(); i++) // fill in right frame
  {
    cur_frame.keypoints_.push_back(px_cur[i]);

    if(status[i] != 0){ // we correctly tracked the point
      nrValidDepths += 1;
    }else{
      ref_frame.landmarks_[i] = -1; // make point invalid
    }
  }

  if(cur_frame.keypoints_.size() != ref_frame.keypoints_.size())
    throw std::runtime_error("computeStereo: error -  length of computeStereo is incorrect");

  std::cout << "stereo matching: matched  " << nrValidDepths <<
      " out of " << ref_frame.keypoints_.size() << " keypoints" << std::endl;
}

/* --------------------------------------------------------------------------------------- */
StatusKeypointsCV StereoFrame::getRightKeypointsRectified(
    const cv::Mat left_rectified, const cv::Mat right_rectified, const StatusKeypointsCV left_keypoints_rectified,
    const double fx, const double baseline) const
{
  int verbosity = 0;
  bool writeImageLeftRightMatching = false;

  // The stripe has to be places in the right image, on the left-hand-side wrt the x of the left feature, since:
  // disparity = left_px.x - right_px.x, hence we check: right_px.x < left_px.x
  // a stripe to select in the right image (this must contain match as epipolar lines are horizontal)
  int stripe_rows = sparseStereoParams_.templ_rows + sparseStereoParams_.stripe_extra_rows; // must be odd; p/m stripe_extra_rows/2 pixels to deal with rectification error
  // dimension of the search space in right camera is defined by min depth:
  // depth = fx * b / disparity => max disparity = fx * b / minDepth;
  int stripe_cols = round(fx * baseline / sparseStereoParams_.minPointDist) + sparseStereoParams_.templ_cols + 4; // 4 is a tolerance
  //std::cout << "stripe_cols " << stripe_cols << " stripe_rows " << stripe_rows << " right_rectified.cols "<< right_rectified.cols
  //    << " minPointDist " << sparseStereoParams_.minPointDist << std::endl;
  if(stripe_cols % 2 != 1) {stripe_cols +=1;} // make it odd, if it is not
  if(stripe_cols > right_rectified.cols) { stripe_cols = right_rectified.cols; } // if we exagerated with the stripe columns

  // for each point in the (rectified) left image we try to get the pixel which maximizes
  // correlation with (rectified) right image along the (horizontal) epipolar line
#ifdef USE_OMP
  Matrixf right_keypoints_rectified_matrix = Matrixf::Zero(2,left_keypoints_rectified.size());
  Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> status_right_keypoints_rectified_matrix =
      Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(1,left_keypoints_rectified.size());
  #pragma omp parallel for
  for (size_t i = 0; i < left_keypoints_rectified.size(); ++i)
  {
    // std::cout << "Using " << omp_get_num_threads() << " parallel cores" << std::endl;
    if(left_keypoints_rectified[i].first != Kstatus::VALID){ // skip invalid points (fill in with placeholders in right)
      continue;
    }

    // Do left->right matching
    KeypointCV left_rectified_i = left_keypoints_rectified[i].second;
    StatusKeypointCV right_rectified_i_candidate; double matchingVal_LR;
    std::tie(right_rectified_i_candidate,matchingVal_LR) = findMatchingKeypointRectified(left_rectified, left_rectified_i, right_rectified,
        sparseStereoParams_.templ_cols, sparseStereoParams_.templ_rows,
        stripe_cols, stripe_rows, sparseStereoParams_.toleranceTemplateMatching, writeImageLeftRightMatching);

    right_keypoints_rectified_matrix(0,i) = right_rectified_i_candidate.second.x;
    right_keypoints_rectified_matrix(1,i) = right_rectified_i_candidate.second.y;
    status_right_keypoints_rectified_matrix(0,i) = right_rectified_i_candidate.first;
  }
  StatusKeypointsCV right_keypoints_rectified; right_keypoints_rectified.reserve(left_keypoints_rectified.size());
  for (size_t i = 0; i < left_keypoints_rectified.size(); ++i){
    if(left_keypoints_rectified[i].first != Kstatus::VALID){
      right_keypoints_rectified.push_back(std::make_pair(left_keypoints_rectified[i].first, KeypointCV(0.0,0.0)));
    }else{
      right_keypoints_rectified.push_back(std::make_pair(static_cast<Kstatus>(status_right_keypoints_rectified_matrix(i)),
          KeypointCV(right_keypoints_rectified_matrix(0,i),right_keypoints_rectified_matrix(1,i))));
    }
  }
#endif
  StatusKeypointsCV right_keypoints_rectified; right_keypoints_rectified.reserve(left_keypoints_rectified.size());

  // Serial version
  for (size_t i = 0; i < left_keypoints_rectified.size(); ++i)
  {
    // check if we already have computed the right kpt, in which case we avoid recomputing
    if(left_keypoints_rectified_.size() > i+1 &&
        right_keypoints_rectified_.size() > i+1 &&
        right_keypoints_status_.size() > i+1 &&  // if we stored enough points
        left_keypoints_rectified[i].second.x == left_keypoints_rectified_[i].x && // the query point matches the one we stored
        left_keypoints_rectified[i].second.y == left_keypoints_rectified_[i].y){
      // we already stored the rectified pixel in the stereo frame
      right_keypoints_rectified.push_back(std::make_pair(right_keypoints_status_[i],right_keypoints_rectified_[i]));
      continue;
    }

    // if the left point is invalid, we also set the right point to be invalid and we move on
    if(left_keypoints_rectified[i].first != Kstatus::VALID){ // skip invalid points (fill in with placeholders in right)
      right_keypoints_rectified.push_back(std::make_pair(left_keypoints_rectified[i].first, KeypointCV(0.0,0.0)));
      continue;
    }

    // Do left->right matching
    KeypointCV left_rectified_i = left_keypoints_rectified[i].second;
    StatusKeypointCV right_rectified_i_candidate; double matchingVal_LR;
    std::tie(right_rectified_i_candidate,matchingVal_LR) = findMatchingKeypointRectified(left_rectified, left_rectified_i, right_rectified,
        sparseStereoParams_.templ_cols, sparseStereoParams_.templ_rows,
        stripe_cols, stripe_rows, sparseStereoParams_.toleranceTemplateMatching, writeImageLeftRightMatching);

    // perform bidirectional check: disabled!
    //if(sparseStereoParams_.bidirectionalMatching && right_rectified_i_candidate.first == Kstatus::VALID){
    //
    //  throw std::runtime_error("getRightKeypointsRectified: bidirectional matching was not updated to deal with small stripe size");
    //  StatusKeypointCV left_rectified_i_candidate; double matchingVal_RL;
    //  std::tie(left_rectified_i_candidate,matchingVal_RL) = findMatchingKeypointRectified(right_rectified, right_rectified_i_candidate.second, left_rectified,
    //      sparseStereoParams_.templ_cols, sparseStereoParams_.templ_rows,
    //      stripe_cols, stripe_rows, sparseStereoParams_.toleranceTemplateMatching);
    //
    //  if(fabs(left_rectified_i_candidate.second.x - left_rectified_i.x) > 5  // if matching is not bidirectional
    //      ||  fabs(left_rectified_i_candidate.second.y - left_rectified_i.y) > 5)
    //   // ||  fabs(matchingVal_LR-matchingVal_RL) > 0.1 * matchingVal_RL ) // and score is not similar in the two directions (found unnecessary)
    //  {
    //    right_rectified_i_candidate.first = Kstatus::NO_RIGHT_RECT;
    //    if(verbosity>0)
    //    {
    //      std::cout << "-------------------------------------" <<std::endl;
    //      std::cout << "matchingVal_LR " << matchingVal_LR <<std::endl;
    //      std::cout << "matchingVal_RL " << matchingVal_RL <<std::endl;
    //      std::cout << "left_rectified_i_candidate " << left_rectified_i_candidate.second <<std::endl;
    //      std::cout << "left_rectified_i " << left_rectified_i <<std::endl;
    //      std::cout << "-------------------------------------" <<std::endl;
    //    }
    //  }
    //}
    right_keypoints_rectified.push_back(right_rectified_i_candidate);
  }

  if(verbosity>0)
  {
    cv::Mat imgL_withKeypoints = UtilsOpenCV::DrawCircles(left_rectified, left_keypoints_rectified);
    cv::Mat imgR_withKeypoints = UtilsOpenCV::DrawCircles(right_rectified, right_keypoints_rectified);
    showImagesSideBySide(imgL_withKeypoints,imgR_withKeypoints,"result_getRightKeypointsRectified",verbosity);
  }
  return right_keypoints_rectified;
}
/* --------------------------------------------------------------------------------------- */
std::pair<StatusKeypointCV,double> StereoFrame::findMatchingKeypointRectified(const cv::Mat left_rectified, const KeypointCV left_rectified_i,
    const cv::Mat right_rectified, const int templ_cols, const int templ_rows, const int stripe_cols, const int stripe_rows,
    const double tol_corr, const bool debugStereoMatching) const
{
  /// correlation matrix
  int result_cols =  stripe_cols - templ_cols + 1;
  int result_rows = stripe_rows - templ_rows + 1;
  cv::Mat result;
  // result.create( result_rows, result_cols, CV_32FC1 );

  int rounded_left_rectified_i_x = round(left_rectified_i.x);
  int rounded_left_rectified_i_y = round(left_rectified_i.y);

  /// CORRECTLY PLACE THE TEMPLATE (IN LEFT IMAGE) ///////////////////////////
  int temp_corner_y = rounded_left_rectified_i_y - (templ_rows-1)/2; // y-component of upper left corner of template
  if(temp_corner_y < 0 || temp_corner_y + templ_rows > left_rectified.rows-1){ // template exceeds bottom or top of the image
    return std::make_pair(std::make_pair(Kstatus::NO_RIGHT_RECT, KeypointCV(0.0,0.0)), -1.0);  // skip point too close to up or down boundary
  }
  int offset_temp = 0; // compensate when the template falls off the image
  int temp_corner_x = rounded_left_rectified_i_x - (templ_cols-1)/2;
  if(temp_corner_x < 0){ // template exceeds on the left-hand-side of the image
    offset_temp = temp_corner_x; // offset_temp a bit to make the template inside the image
    temp_corner_x = 0; // because of the offset_temp, the template corner ends up on the image border
  }
  if(temp_corner_x + templ_cols > left_rectified.cols-1){ // template exceeds on the right-hand-side of the image
    if(offset_temp != 0)
      throw std::runtime_error("findMatchingKeypointRectified: offset_temp cannot exceed in both directions!");
    offset_temp = (temp_corner_x + templ_cols) - ( left_rectified.cols - 1 ); // amount that exceeds
    temp_corner_x -= offset_temp; // corner has to be offset_temp to the left by the amount that exceeds
  }
  // create template
  cv::Rect templ_selector = cv::Rect(temp_corner_x, temp_corner_y, templ_cols, templ_rows);
  cv::Mat templ(left_rectified, templ_selector);
  //templ.convertTo(templ, CV_8U);

  // std::cout << "template: \n" << templ << std::endl;

  /// CORRECTLY PLACE THE STRIPE (IN RIGHT IMAGE) ///////////////////////////
  int stripe_corner_y = rounded_left_rectified_i_y - (stripe_rows-1)/2; // y-component of upper left corner of stripe
  if(stripe_corner_y < 0 || stripe_corner_y + stripe_rows > right_rectified.rows-1){ // stripe exceeds bottom or top of the image
    return std::make_pair(std::make_pair(Kstatus::NO_RIGHT_RECT, KeypointCV(0.0,0.0)), -1.0);  // skip point too close to boundary
  }
  int offset_stripe = 0; // compensate when the template falls off the image
  int stripe_corner_x = rounded_left_rectified_i_x + (templ_cols-1)/2 - stripe_cols; // y-component of upper left corner of stripe
  if(stripe_corner_x + stripe_cols > right_rectified.cols-1){ // stripe exceeds on the right of image
    offset_stripe = (stripe_corner_x + stripe_cols) - (right_rectified.cols - 1); // amount that exceeds
    stripe_corner_x -= offset_stripe;
  }
  if(stripe_corner_x < 0) // stripe exceeds on the left of the image
    stripe_corner_x = 0; // set to left-most column
  // create stripe
  cv::Rect stripe_selector = cv::Rect(stripe_corner_x, stripe_corner_y, stripe_cols, stripe_rows);
  cv::Mat stripe(right_rectified, stripe_selector);
  //stripe.convertTo(stripe, CV_8U);

  // find template and normalize results
  double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;

  cv::matchTemplate( stripe, templ, result, CV_TM_SQDIFF_NORMED );
  //result.convertTo(result, CV_32F);
  /// Localizing the best match with minMaxLoc
  cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat() );

  // normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() ); // TODO: do we need to normalize??

  cv::Point matchLoc = minLoc; // position within the result matrix
  matchLoc.x += stripe_corner_x + (templ_cols-1)/2 + offset_temp;
  matchLoc.y += stripe_corner_y + (templ_rows-1)/2; // from result to image
  KeypointCV match_px = KeypointCV(matchLoc.x,matchLoc.y); // our desired pixel match

  // debug:
  //int result_cols =  stripe_cols - templ_cols + 1;
  //int result_rows = stripe_rows - templ_rows + 1;
  //  if(debugStereoMatching){
  //    KeypointsCV left_rectified_i_draw; left_rectified_i_draw.push_back(left_rectified_i);
  //    cv::Mat left_rectifiedWithKeypoints = UtilsOpenCV::DrawCircles(left_rectified, left_rectified_i_draw);
  //    std::cout << "Rectangle 1" << std::endl;
  //    rectangle( left_rectifiedWithKeypoints, templ_selector, cv::Scalar( 0, 255, 255 ));
  //    cv::Mat right_rectified_rect;
  //    right_rectified.copyTo(right_rectified_rect);
  //    cv::cvtColor(right_rectified_rect, right_rectified_rect, cv::COLOR_GRAY2BGR);
  //    std::cout << "converted to color" << std::endl;
  //    cv::circle(right_rectified_rect, matchLoc, 3, cv::Scalar(0, 0, 255), 2);
  //    cv::Point textLoc = matchLoc + cv::Point(-10,-5);
  //    cv::putText(right_rectified_rect, UtilsOpenCV::To_string_with_precision(minVal),
  //        textLoc, CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 255));
  //    std::cout << "put text" << std::endl;
  //    rectangle( right_rectified_rect, stripe_selector, cv::Scalar( 0, 255, 255 ));
  //    std::cout << "Rectangle 2" << std::endl;
  //    std::string img_title = "rectifiedWithKeypointsAndRects_" + std::to_string(std::rand()) + "_" ;
  //    std::cout << img_title << std::endl;
  //    showImagesSideBySide(left_rectifiedWithKeypoints,right_rectified_rect,img_title, 2);
  //  }

  // refine keypoint with subpixel accuracy
  if(sparseStereoParams_.subpixelRefinement){
    cv::TermCriteria criteria = cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
    cv::Size winSize = cv::Size(10, 10);
    cv::Size zeroZone = cv::Size(-1, -1);
    std::vector<cv::Point2f> corner; corner.push_back(match_px); // make in correct format for cornerSubPix
    cv::cornerSubPix(right_rectified, corner, winSize, zeroZone, criteria);
    match_px = corner[0];
  }

  if(minVal < tol_corr){ // Valid point with small mismatch wrt template
    return std::make_pair(std::make_pair(Kstatus::VALID, match_px), minVal);
  }else{
    return std::make_pair(std::make_pair(Kstatus::NO_RIGHT_RECT, match_px), minVal);
  }
}
/* --------------------------------------------------------------------------------------- */
std::vector<double> StereoFrame::getDepthFromRectifiedMatches(StatusKeypointsCV& left_keypoints_rectified,
    StatusKeypointsCV& right_keypoints_rectified, const double fx, const double b) const
{
  // depth = fx * baseline / disparity (should be fx = focal * sensorsize)
  double fx_b = fx * b;

  std::vector<double> disparities, depths;
  if(left_keypoints_rectified.size() != right_keypoints_rectified.size())
    throw std::runtime_error("getDepthFromRectifiedMatches: size mismatch!");

  int nrValidDepths = 0;
  // disparity = left_px.x - right_px.x, hence we check: right_px.x < left_px.x
  for(size_t i=0; i<left_keypoints_rectified.size(); i++){
    if (left_keypoints_rectified[i].first == Kstatus::VALID && right_keypoints_rectified[i].first == Kstatus::VALID)
    {
      KeypointCV left_px = left_keypoints_rectified[i].second;
      KeypointCV right_px = right_keypoints_rectified[i].second;
      double disparity = left_px.x - right_px.x;
      if(disparity >= 0) // valid
      {
        nrValidDepths += 1;
        double depth = fx_b / disparity;
        // std::cout << "depth " << depth << std::endl;
        if(depth < sparseStereoParams_.minPointDist || depth > sparseStereoParams_.maxPointDist){
          right_keypoints_rectified[i].first = Kstatus::NO_DEPTH;
          depths.push_back(0.0);
        }else{
          depths.push_back(depth);
        }
      }else{ // right match was wrong
        right_keypoints_rectified[i].first = Kstatus::NO_DEPTH;
        depths.push_back(0.0);
      }
    }else{ // something is wrong
      if(left_keypoints_rectified[i].first != Kstatus::VALID){
        // we cannot have a valid right, without a valid left keypoint
        right_keypoints_rectified[i].first = left_keypoints_rectified[i].first;
      }
      depths.push_back(0.0);
    }
  }
  if(left_keypoints_rectified.size() != depths.size())
    throw std::runtime_error("getDepthFromRectifiedMatches: depths size mismatch!");

  return depths;
}