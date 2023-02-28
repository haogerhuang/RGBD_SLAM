#include "../include/feature_helper.h"
#include "../include/utils.h"


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
){


    for (size_t i = 0; i < knn_match.size(); i++){
        if (knn_match[i][0].distance < ratio * knn_match[i][1].distance){
            good_match.push_back(knn_match[i][0]);

            cv::Point2f p1 = keypoints1[knn_match[i][0].queryIdx].pt;
            cv::Point2f p2 = keypoints2[knn_match[i][0].trainIdx].pt;

            pnts1.push_back(Eigen::Vector3f(p1.x, p1.y, 1));
            pnts2.push_back(Eigen::Vector3f(p2.x, p2.y, 1));

            filter_keypoints1.push_back(keypoints1[knn_match[i][0].queryIdx]);
            filter_keypoints2.push_back(keypoints2[knn_match[i][0].trainIdx]);

            des_indexs1.push_back(knn_match[i][0].queryIdx);
            des_indexs2.push_back(knn_match[i][0].trainIdx);
        }
    }


}


