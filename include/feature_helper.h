#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <opencv2/opencv.hpp>
#include "utils.h"


void filter_matches(
    const std::vector<std::vector<cv::DMatch>>& knn_match,
    std::vector<cv::DMatch> &good_match,
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2,
    std::vector<cv::KeyPoint> &filter_keypoints1,
    std::vector<cv::KeyPoint> &filter_keypoints2,
    std::vector<int> &des_indexs1,
    std::vector<int> &des_indexs2,
    VecVector3f& pnts1,
    VecVector3f& pnts2,
    const float ratio
);


#endif
