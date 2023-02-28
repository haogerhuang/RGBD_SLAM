#ifndef SLAM_H
#define SLAM_H

#include "rgbd.h"
#include <sophus/se3.hpp>

#include <pcl/common/angles.h> // for pcl::deg2rad
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class Slam{
    public:
        Slam(){
            cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        }
        void insert(RGBD data);
        void BA(int iters);
        void BA1(int iters);
        void BA2(int iters);
        void BA_dense(int iters, int s);
        void BA_sparse(int iters);
        void visualize_all();
        void visualize_kf();
        void visualize_wp();
        
        std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>> cloud;
    private:
        void init_pose(RGBD& data);
        void update_placeholder(RGBD& data);
        Sophus::SE3f keypoint_filter(VecVector3f &pnts1, VecVector3f &pnts2);
        
        void local_matching_with_global(RGBD& cur_rgbd, VecVector3f& cur_pnts, VecVector3f& matched_global_pnts);
        void local_matching_with_kf(RGBD& cur_rgbd, std::vector<int>& selected_kf, VecVector3f& cur_pnts, VecVector3f& matched_global_pnts);
        void local_dense(RGBD& data);
        void local_dense1(RGBD& data);
        void GN_sparse(const VecVector3f& pnts1, Sophus::SE3f& pose1, 
                     const VecVector3f& pnts2, Sophus::SE3f& pose2);

        bool is_kf(RGBD& rgbd, int s);

        float accu_jacobian_sparse(const VecVector3f& pnts1, const VecVector3f& pnts2, 
                                  Sophus::SE3f& T, Eigen::MatrixXf& H, Eigen::VectorXf& b);

        float accu_jacobian_dense(RGBD& cur_rgbd, RGBD& prev_rgbd, Sophus::SE3f& T1, Sophus::SE3f& T2, Eigen::MatrixXf& H, Eigen::VectorXf& b);
        float accu_jacobian_dense_from_world(RGBD& cur_rgbd, Sophus::SE3f& T, Eigen::MatrixXf& H, Eigen::VectorXf& b);
        float accu_jacobian_dense_from_kf(RGBD& cur_rgbd, std::vector<int>& selected_kf, Sophus::SE3f& T, Eigen::MatrixXf& H, Eigen::VectorXf& b, int s, std::vector<float>& dense_err);
        void GN(RGBD& rgbd);

        void ICP(RGBD& rgbd1, RGBD& rgbd2);
        void add_points(RGBD& rgbd);
        void add_points1(RGBD& rgbd);
        void add_kf_kps(RGBD& rgbd);

        std::vector<int> find_overlap_kfs(RGBD& rgbd, int s, int num);

        Eigen::VectorXf GN_solve(Eigen::VectorXf H_ll,
                    Eigen::MatrixXf H_pl,
                    Eigen::MatrixXf H_pp,
                    Eigen::VectorXf b_l,
                    Eigen::VectorXf b_p,
                    float damping_factor);

        void optical_flow(RGBD& rgbd);

        std::vector<RGBD> rgbd_list;
        // std::vector<int> kf_list;
        VecVector3f world_pnts;
        VecVector3f world_rgbs;

        VecVector3f local_match_pnts1;
        VecVector3f local_match_pnts2;

        int kf_interval = 10;

        std::vector<RGBD> tmp_rgbd_list; // tmp place holder
        Sophus::SE3f vel;
        Sophus::SE3f last_pose;

        int num_frames = 0;

        int local_match_num = 50;

        // Same size as world_pnts
        // graph[i] -> {keyframe_idx -> keypoint idx}
        std::vector<std::map<int,int>> graph;

        cv::Mat global_descriptors;
        std::vector<cv::KeyPoint> global_keypoints;
        std::vector<std::vector<std::pair<int,int>>> kp_global2local;
        std::vector<Eigen::Vector3f> global_kp_3d;

        std::vector<int> local_unmatch_idx;

        std::vector<int> cur_selected_kfs;
        
};

#endif
