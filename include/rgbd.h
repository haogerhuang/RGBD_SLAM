#ifndef RGBD_H
#define RGBD_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include "utils.h"


class RGBD{
    public:
    RGBD(std::string name, int idx, std::vector<float> _scales, float depth_factor_){
        frame_idx = idx;

        depth_factor = depth_factor_;

        std::string id_str = std::to_string(idx);
        std::string pad_str(6-id_str.length(), '0');
        std::string img_name = name + "/results/frame" + pad_str + id_str + ".jpg";
        std::string depth_name = name + "/results/depth" + pad_str + id_str + ".png";

        // cout<<img_name<<"\n";

        // std::string img_name = name + "/rgb/frame_" + std::to_string(idx) + ".png";
        // std::string depth_name = name + "/depth/frame_" + std::to_string(idx) + ".png";
        cv::Mat img = cv::imread(img_name, cv::IMREAD_COLOR);

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::Mat depth = cv::imread(depth_name, cv::IMREAD_UNCHANGED);

        scales = _scales;

        for(float s: scales){
            cv::Mat tmp_img;
            cv::Mat tmp_depth;

            cv::resize(img, tmp_img, cv::Size(), s, s);
            cv::resize(depth, tmp_depth, cv::Size(), s, s);
            img_list.push_back(tmp_img);
            depth_list.push_back(tmp_depth);
        }
    }
    std::vector<cv::KeyPoint> get_keypoints();
    Eigen::Vector3f get_kp_by_id(int i);
    VecVector6f get_pc(int s);
    VecVector3f get_XYZ(int s);
    cv::Mat get_descriptors();
    Eigen::Vector3f get_xyz(int x, int y, int s);
    Eigen::Vector3f get_rgb(int x, int y, int s);
    float get_intensity(int x, int y, int s);
    void setK(Eigen::Matrix3f _K);
    float get_fx(int s);
    float get_fy(int s);
    int get_rows(int s);
    int get_cols(int s);
    Eigen::Matrix3f getK(int s);
    cv::Mat get_img(int s);
    cv::Mat get_depth(int s);
    cv::Mat get_gray(int s);
    Sophus::SE3f get_pose();
    void set_pose(Sophus::SE3f pose_);

    std::pair<cv::Mat, cv::Mat> get_img_gradient();
    std::pair<cv::Mat, cv::Mat> get_depth_gradient();
    cv::Mat get_normal();
    float depth_factor;

    int frame_idx;
    std::vector<int> kp2world;
    VecVector3f kp_3d;

    bool correct_pose = true;

    private:
    std::vector<float> scales;
    

    std::vector<Eigen::Matrix3f> K_list;
    std::vector<cv::Mat> img_list;
    std::vector<cv::Mat> depth_list;

    std::pair<cv::Mat, cv::Mat> img_gradient;
    std::pair<cv::Mat, cv::Mat> depth_gradient;
    cv::Mat normal;

    bool need_GN_data = true;
    
    //Eigen::Matrix3d K;
    //Cv::Mat img;
    //Cv::Mat depth;

    //Eigen::Matrix3d down_K;
    //Cv::Mat down_img;
    //Cv::Mat down_gray_img;
    //Cv::Mat down_depth;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    VecVector6f point_cloud;
    VecVector3f point_cloud_xyz;
    Sophus::SE3f pose;

    

    void compute_GN_data(int s);

    void compute_Gradient(cv::Mat &img, cv::Mat &gradient_x, cv::Mat &gradient_y);
    void compute_Normal(cv::Mat& gradient_x, cv::Mat& gradient_y);
};

#endif
