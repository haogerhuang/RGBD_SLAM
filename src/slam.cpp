#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Dense>
#include <Eigen/LU>

#include <math.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <queue>

#include "../include/utils.h"
#include "../include/feature_helper.h"
#include "../include/slam.h"

#include <pcl/common/angles.h> // for pcl::deg2rad
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>


void Slam::insert(RGBD data){

    init_pose(data);
    
    if (!rgbd_list.empty()){
        GN(data);   
    }
    
    if((num_frames % kf_interval == 0) && data.correct_pose){
        rgbd_list.push_back(data);
        add_points(data);
      
        // std::cout<<"IS KEYFRAME!!!!!!!!!!!!!! Current number of kfs: "<<rgbd_list.size()<<"\n";
       
    }
    // rgbd_list.push_back(data);
    if (data.correct_pose){
        update_placeholder(data);
        last_pose = data.get_pose();

        num_frames++;
    }
    // cout<<rgbd_list.back().get_pose().matrix()<<"\n";
}

void Slam::init_pose(RGBD& data){
  
    if(tmp_rgbd_list.size() >= 2)
        vel = tmp_rgbd_list[1].get_pose() * tmp_rgbd_list[0].get_pose().inverse();
    // data.set_pose(vel * last_pose);
    data.set_pose(last_pose);
}

void Slam::update_placeholder(RGBD& data){
    if(tmp_rgbd_list.size()==2){
        tmp_rgbd_list[0] = tmp_rgbd_list[1];
        tmp_rgbd_list[1] = data;
    }
    else{
        tmp_rgbd_list.push_back(data);
    }
}

Sophus::SE3f Slam::keypoint_filter(
    VecVector3f &pnts1,
    VecVector3f &pnts2
){
    VecVector3f filter_pnts1;
    VecVector3f filter_pnts2;
    Sophus::SE3f T12;
    std::cout<<"filter "<<pnts1.size()<<"\n";
    while (true){
        T12 = kabsh_algorithm(pnts1, pnts2);
        float ttl_err = 0.0;
    
        for (size_t i = 0; i < pnts1.size(); i++){
            Eigen::Vector3f pnt1 = pnts1[i];
            Eigen::Vector3f pnt2 = pnts2[i];
            pnt1 = T12 * pnt1;
            float error = std::sqrt((pnt1[0]-pnt2[0])*(pnt1[0]-pnt2[0]) +\
                                  (pnt1[1]-pnt2[1])*(pnt1[1]-pnt2[1]) +\
                                  (pnt1[2]-pnt2[2])*(pnt1[2]-pnt2[2]));
            if (error <= 0.05){
                filter_pnts1.push_back(pnt1);
                filter_pnts2.push_back(pnt2);
            }
            ttl_err += error;
        }
        cout<<filter_pnts1.size()<<" "<<ttl_err<<"\n";
        if (filter_pnts1.size() == pnts1.size()) break;
        pnts1 = filter_pnts1;
        pnts2 = filter_pnts2;
        filter_pnts1 = {};
        filter_pnts2 = {};

    }
    return T12;
}

void Slam::local_matching_with_global(RGBD& cur_rgbd, VecVector3f& cur_pnts, VecVector3f& matched_global_pnts){

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();

    std::vector<cv::KeyPoint> cur_keypoints = cur_rgbd.get_keypoints(); 
    cv::Mat cur_descriptors = cur_rgbd.get_descriptors();

    std::vector<std::vector<cv::DMatch>> knn_match;
    matcher->knnMatch(cur_descriptors, global_descriptors, knn_match, 2);
                
    VecVector3f local_pnts;
    VecVector3f global_pnts;

    std::vector<int> cur_indexs; //no use
    std::vector<int> global_indexs; //no use

    std::vector<cv::KeyPoint> filter_cur_kp;
    std::vector<cv::KeyPoint> filter_global_kp;

    std::vector<cv::DMatch> good_match;
    filter_matches(knn_match, 
                 good_match,
                 cur_keypoints, global_keypoints, 
                 filter_cur_kp, filter_global_kp,
                 cur_indexs, global_indexs,
                 local_pnts, global_pnts, 0.6);
    

    //Transform points to 3D
    for (int k = 0; k < local_pnts.size(); k++){
        int x = (int)local_pnts[k][0];
        int y = (int)local_pnts[k][1];

        Eigen::Vector3f xyz = cur_rgbd.get_xyz(x, y, 0);

        cur_pnts.push_back(xyz);
        matched_global_pnts.push_back(global_kp_3d[global_indexs[k]]);
    }

    // if(cur_pnts.size() > local_match_num) break;
    std::set<int> local_index;
    for(int idx : cur_indexs) local_index.insert(idx);

    local_unmatch_idx = {};
    for(int i = 0; i < cur_keypoints.size(); i++){
        if(!local_index.empty() && (i == *local_index.begin())){
            local_index.erase(local_index.begin());  
        }
        else local_unmatch_idx.push_back(i);
    }
    



}

void Slam::local_matching_with_kf(RGBD& cur_rgbd, std::vector<int>& selected_kf, VecVector3f& cur_pnts, VecVector3f& matched_global_pnts){

    std::vector<RGBD*> local_rgbd_list;
    for(int i: selected_kf){
        local_rgbd_list.push_back(&rgbd_list[i]);
    }

    if (num_frames - 1 % kf_interval != 0){
        local_rgbd_list.push_back(&tmp_rgbd_list.back());
    }

    for(RGBD* prev_rgbd: local_rgbd_list){
    //for(int i = std::max(0, rgbd_list.size()-5); i < rgbd_list.size(); i++){

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();


        std::vector<cv::KeyPoint> keypoints1 = prev_rgbd->get_keypoints(); 
        cv::Mat descriptors1 = prev_rgbd->get_descriptors();

        std::vector<cv::KeyPoint> keypoints2 = cur_rgbd.get_keypoints(); 
        cv::Mat descriptors2 = cur_rgbd.get_descriptors();

        std::vector<std::vector<cv::DMatch>> knn_match;
        matcher->knnMatch(descriptors1, descriptors2, knn_match, 2);
                    
        VecVector3f pnts1;
        VecVector3f pnts2;

        std::vector<int> des_indexs1; //no use
        std::vector<int> des_indexs2; //no use

        std::vector<cv::KeyPoint> filter_keypoints1;
        std::vector<cv::KeyPoint> filter_keypoints2;

        std::vector<cv::DMatch> good_match;
        filter_matches(knn_match, 
                    good_match,
                    keypoints1, keypoints2, 
                    filter_keypoints1, filter_keypoints2,
                    des_indexs1, des_indexs2,
                    pnts1, pnts2, 0.6);

        
        if (pnts1.size() < 50) continue;
        // VecVector3f init_pnts1;
        // VecVector3f init_pnts2;
        
        Sophus::SE3f kf_pose = prev_rgbd->get_pose();

        //Transform points to 3D
        for (int k = 0; k < pnts1.size(); k++){
            int x1 = (int)pnts1[k][0];
            int y1 = (int)pnts1[k][1];
            int x2 = (int)pnts2[k][0];
            int y2 = (int)pnts2[k][1];
            Eigen::Vector3f xyz1 = prev_rgbd->get_xyz(x1, y1, 0);
            Eigen::Vector3f xyz2 = cur_rgbd.get_xyz(x2, y2, 0);

            // init_pnts1.push_back(xyz1);
            // init_pnts2.push_back(xyz2);
            // local_match_pnts1.push_back(xyz2);
            // local_match_pnts2.push_back(xyz1);
            cur_pnts.push_back(xyz2);
            matched_global_pnts.push_back(kf_pose * xyz1);
        }

        // if(cur_pnts.size() > local_match_num) break;

    }

}

void Slam::add_kf_kps(RGBD& cur_rgbd){
    Sophus::SE3f pose = cur_rgbd.get_pose();
    if(local_unmatch_idx.empty()){
        this->global_descriptors = cur_rgbd.get_descriptors();
        global_keypoints = cur_rgbd.get_keypoints();
        for(int i = 0; i < global_keypoints.size(); i++){  
            cv::KeyPoint kp = global_keypoints[i];
            Eigen::Vector3f pnt = cur_rgbd.get_xyz(kp.pt.x, kp.pt.y, 0);
            global_kp_3d.push_back(pose * pnt);
        }
    } 
    else{
        std::vector<cv::KeyPoint> cur_kps = cur_rgbd.get_keypoints();
        cv::Mat cur_des = cur_rgbd.get_descriptors();
        for(int idx: local_unmatch_idx){
            global_keypoints.push_back(cur_kps[idx]); 
            cv::vconcat(global_descriptors, cur_des.row(idx), global_descriptors);
            Eigen::Vector3f pnt = cur_rgbd.get_xyz(cur_kps[idx].pt.x, 
                                                   cur_kps[idx].pt.y, 0);
            global_kp_3d.push_back(pose * pnt);
        }
    }
}

std::vector<int> Slam::find_overlap_kfs(RGBD& cur_rgbd, int s, int num){
    
    int h = cur_rgbd.get_rows(s);
    int w = cur_rgbd.get_cols(s);
    Eigen::Matrix3f K = cur_rgbd.getK(s);
    float fx = K(0,0);
    float fy = K(1,1);

    cv::Mat cur_img = cur_rgbd.get_img(s);
    cv::Mat cur_depth = cur_rgbd.get_depth(s);

    Sophus::SE3f cur_pose = cur_rgbd.get_pose();

    int pad = 1;
    std::vector<int> x_offsets = {pad, w-1-pad};
    std::vector<int> y_offsets = {pad, h-1-pad};

    std::vector<int> init_match_kfs;

    for(int i = 0; i < rgbd_list.size(); i++){
        RGBD kf_rgbd = rgbd_list[i]; 

        int cnt = 0;
        for(int x : x_offsets){
            for(int y: y_offsets){
                Eigen::Vector3f xyz = cur_rgbd.get_xyz(x, y, s);
                Eigen::Vector3f rgb = cur_rgbd.get_rgb(x, y, s);
                
                xyz = cur_pose * xyz;

                Eigen::Vector3f proj_xyz = kf_rgbd.get_pose().inverse() * xyz;
                proj_xyz = cam2pixel(proj_xyz/proj_xyz[2], K);
                Eigen::Vector3f proj_rgb = kf_rgbd.get_rgb(proj_xyz[0], proj_xyz[1], s);

                if(proj_rgb[2] < 0) continue;

                proj_xyz = kf_rgbd.get_xyz(proj_xyz[0], proj_xyz[1], s);
                proj_xyz = kf_rgbd.get_pose() * proj_xyz;

                if((xyz - proj_xyz).norm() < 0.5 && (rgb - proj_rgb).norm() < 50){
                    cnt++;
                }
            
            }
        }
        if (cnt > 0) init_match_kfs.push_back(i);
        
    }

    //std::vector<int> kf_match_cnt(rgbd_list.size(), 0);
    std::vector<int> kf_match_cnt(init_match_kfs.size(), 0);

    int match_cnt = 0;

    for(int x = 0; x < w; x++){
        for(int y = 0; y < h; y++){
            Eigen::Vector3f xyz = cur_rgbd.get_xyz(x, y, s);
            Eigen::Vector3f rgb = cur_rgbd.get_rgb(x, y, s);

            xyz = cur_pose * xyz;

            //for(int kf_i = 0; kf_i < rgbd_list.size(); kf_i++){
            for(int i = 0; i < init_match_kfs.size(); i++){
                int kf_i = init_match_kfs[i];
                RGBD kf_rgbd = rgbd_list[kf_i];
                Eigen::Vector3f proj_xyz = kf_rgbd.get_pose().inverse() * xyz;
                proj_xyz = cam2pixel(proj_xyz/proj_xyz[2], K);
                if (proj_xyz[0] < pad || proj_xyz[0] > w-pad-1 || proj_xyz[1] < pad 
                    || proj_xyz[1] > h-pad-1) continue;
                Eigen::Vector3f proj_rgb = kf_rgbd.get_rgb(proj_xyz[0], proj_xyz[1], s);
                proj_xyz = kf_rgbd.get_xyz(proj_xyz[0], proj_xyz[1], s);
                proj_xyz = kf_rgbd.get_pose() * proj_xyz;
                if((xyz - proj_xyz).norm() < 0.1 && (rgb - proj_rgb).norm() < 30){
                    kf_match_cnt[i]++;
                }
            }
        }
    }
    std::priority_queue<std::pair<int,int>> pq;
    for(int i = 0; i < kf_match_cnt.size(); i++){   
        pq.push({kf_match_cnt[i], init_match_kfs[i]});
    }
    std::vector<int> select_kf;
    while(!pq.empty() && select_kf.size() < num){
        std::pair<int,int> cur = pq.top();
        pq.pop();
        select_kf.push_back(cur.second);
    }
    return select_kf;
}

bool Slam::is_kf(RGBD& cur_rgbd, int s){
    int h = cur_rgbd.get_rows(s);
    int w = cur_rgbd.get_cols(s);
    Eigen::Matrix3f K = cur_rgbd.getK(s);
    float fx = K(0,0);
    float fy = K(1,1);

    cv::Mat cur_img = cur_rgbd.get_img(s);
    cv::Mat cur_depth = cur_rgbd.get_depth(s);

    Sophus::SE3f cur_pose = cur_rgbd.get_pose();

    int match_cnt = 0;
    for(int x = 0; x < w; x++){
        for(int y = 0; y < h; y++){
            Eigen::Vector3f xyz = cur_rgbd.get_xyz(x, y, s);
            Eigen::Vector3f rgb = cur_rgbd.get_rgb(x, y, s);

            xyz = cur_pose * xyz;

            bool found_match = false;
            for(RGBD kf_rgbd: rgbd_list){
                Eigen::Vector3f proj_xyz = kf_rgbd.get_pose().inverse() * xyz;
                proj_xyz = cam2pixel(proj_xyz/proj_xyz[2], K);
                Eigen::Vector3f proj_rgb = kf_rgbd.get_rgb(proj_xyz[0], proj_xyz[1], s);
                proj_xyz = kf_rgbd.get_xyz(proj_xyz[0], proj_xyz[1], s);
                proj_xyz = kf_rgbd.get_pose() * proj_xyz;
                if((xyz - proj_xyz).norm() < 0.1 && (rgb - proj_rgb).norm() < 30){
                    match_cnt++;
                    break;
                }
            }
        }
    }
    // for(int i = 0; i < world_pnts.size(); i++){
    //     Eigen::Vector3f xyz = world_pnts[i];
    //     Eigen::Vector3f rgb = world_rgbs[i];
    //     xyz = cur_pose * xyz;
        
    // }

    float match_ratio = (float)match_cnt/(float)(h*w);
    std::cout<<"match_ratio"<<match_ratio<<"\n";
    return match_ratio < 0.7;
    
}

void Slam::GN(RGBD& rgbd){
    
    Sophus::SE3f T1 = rgbd.get_pose();

    float prev_err = std::numeric_limits<float>::max();

    float damping_factor = 1.0;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> cur_match_kfs = find_overlap_kfs(rgbd, 2, 5);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);

    std::cout<<"find overlap time: "<<duration.count() * 1e-6 <<"\n";

    cur_selected_kfs = cur_match_kfs;

    VecVector3f cur_pnts;
    VecVector3f global_matched_pnts;

    start = std::chrono::high_resolution_clock::now();
    //local_matching_with_global(rgbd, cur_pnts, global_matched_pnts);
    local_matching_with_kf(rgbd, cur_match_kfs, cur_pnts, global_matched_pnts);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);

    std::cout<<"sparse match time: "<<duration.count() * 1e-6 <<"\n";

    std::vector<float> sparse_errs;

    for(int iter = 0; iter < 10 && cur_pnts.size() > local_match_num; iter++){
        Eigen::MatrixXf H = Eigen::MatrixXf::Zero(6,6);
        Eigen::VectorXf b = Eigen::VectorXf::Zero(6);
        

        float cur_err = accu_jacobian_sparse(cur_pnts, global_matched_pnts, T1, H, b);

        if(sparse_errs.size() < 2) sparse_errs.push_back(cur_err);
        else sparse_errs.back() = cur_err;

        
        Eigen::MatrixXf damping = Eigen::MatrixXf::Identity(6,6);
        H += damping * damping_factor;
        Eigen::VectorXf dx = H.ldlt().solve(b);

        T1 = Sophus::SE3f::exp(dx) * T1;

        // if(prev_err/cur_err < 1 && prev_err/cur_err > 0.99) break;
        if(cur_err/prev_err <= 1 && cur_err/prev_err > 0.99) break;


        if(cur_err < prev_err){
            damping_factor *= 0.1;
        }
        else damping_factor *= 10;


        prev_err = cur_err;
    }

    if (sparse_errs[0] < sparse_errs[1]){
        //T1 = tmp_rgbd_list[1].get_pose();
        init_pose(rgbd);
        T1 = rgbd.get_pose();
    }
    rgbd.set_pose(T1);

    Sophus::SE3f T1_inv = T1.inverse();

    prev_err = std::numeric_limits<float>::max();

    damping_factor = 0.1;

    int s = 1;

    Sophus::SE3f T1_inv_cache = T1_inv;

    std::vector<std::vector<float>> dense_errs;

    for(int iter = 0; iter < 10; iter++){
        Eigen::MatrixXf H = Eigen::MatrixXf::Zero(6,6);
        Eigen::VectorXf b = Eigen::VectorXf::Zero(6);

        std::vector<float> dense_err_iter;

        float cur_err = accu_jacobian_dense_from_kf(rgbd, cur_match_kfs, T1_inv, H, b, s, dense_err_iter);

        if(dense_errs.size() < 2) dense_errs.push_back(dense_err_iter);
        else dense_errs.back() = dense_err_iter;
        
        Eigen::MatrixXf damping = Eigen::MatrixXf::Identity(6,6);
        H += damping * damping_factor;
        Eigen::VectorXf dx = H.ldlt().solve(b);

        T1_inv = Sophus::SE3f::exp(dx) * T1_inv;

        if(cur_err/prev_err < 1 && cur_err/prev_err > 0.99) break;

        if(cur_err < prev_err){
            damping_factor *= 0.1;
        }
        else damping_factor *= 10;
        prev_err = cur_err;
    }

    int err_cnt = 0;
    for(int i = 0; i < dense_errs[0].size(); i++){
        if(dense_errs[0][i] > dense_errs[1][i]) err_cnt++;
    }

    for(float e: dense_errs[0]) std::cout<<e<<" ";
    std::cout<<"\n";
    for(float e: dense_errs[1]) std::cout<<e<<" ";
    std::cout<<"\n";

    if(err_cnt < dense_errs[0].size()/2){
        std::cout<<"Dont Update\n";
        T1_inv = T1_inv_cache;
        int valid_cnt = 0;
        float mean = 0.0;
        for(float e: dense_errs[0]){
            if(e > 0) valid_cnt++;
            mean += e;
        }
        if(mean/valid_cnt > 0.04) rgbd.correct_pose = false;
    }
    else{
        T1_inv_cache = T1_inv;
        int valid_cnt = 0;
        float mean = 0.0;
        for(float e: dense_errs[1]){
            if(e > 0) valid_cnt++;
            mean += e;
        }
        if(mean/valid_cnt > 0.04) rgbd.correct_pose = false;
    }
    
    rgbd.set_pose(T1_inv.inverse());


}

float Slam::accu_jacobian_sparse(const VecVector3f& pnts1, const VecVector3f& pnts2, 
                                 Sophus::SE3f& T, Eigen::MatrixXf& H, Eigen::VectorXf& b){

                                    

    Eigen::MatrixXf Jacobian = Eigen::MatrixXf::Zero(pnts1.size()*3, 6);
    Eigen::VectorXf Residual = Eigen::VectorXf::Zero(pnts1.size()*3);
    
    float cost = 0;
    float prev_error = std::numeric_limits<float>::max();

    for(int i = 0; i < pnts1.size(); i++){
        Eigen::Vector3f pnt1 = pnts1[i];
        Eigen::Vector3f pnt2 = pnts2[i];

        pnt1 = T * pnt1;
        // pnt2 = T2 * pnt2;
        // pnt2 = pose2 * pnt2;

        Eigen::Vector3f e = pnt1 - pnt2;
        
        
        cost += e.norm();

        Eigen::MatrixXf J_i(3, 6);
        J_i.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();
        J_i.block(0, 3, 3, 3) = -Sophus::SO3f::hat(pnt1);

        // Jacobian.row(i) = 2.0 * e.transpose() * J_i;

        Jacobian.block<3,3>(i*3, 0) = Eigen::Matrix3f::Identity();
        Jacobian.block<3,3>(i*3, 3) = -Sophus::SO3f::hat(pnt1);

        Residual.segment(i*3, 3) = e;
        // Residual[i] = e.norm() * e.norm();
    
    }
    // Jacobian /= pnts1.size();
    // Residual /= pnts1.size();

    // Eigen::MatrixXf H = Jacobian.transpose() * Jacobian;
    // Eigen::VectorXf g = -Jacobian.transpose() * Residual;
    H += Jacobian.transpose() * Jacobian;
    b += -Jacobian.transpose() * Residual;

    // Eigen::VectorXf dx = H.ldlt().solve(g);
    // pose1 = Sophus::SE3f::exp(dx) * pose1;
    cout<<"Sparse error: "<<cost/pnts1.size()<<"\n";
        // if (cost <= prev_error && cost / prev_error > 0.999) break;
        // prev_error = cost;
        
        // if(iter == 0 || iter == ttl_iter-1)
		//     cout<<cost<<endl;
        
	// }
    return cost/pnts1.size();
}

float Slam::accu_jacobian_dense(RGBD& cur_rgbd, RGBD& prev_rgbd, Sophus::SE3f& T1, Sophus::SE3f& T2, Eigen::MatrixXf& H, Eigen::VectorXf& b){

    int s = 1;
    int h = cur_rgbd.get_rows(s);
    int w = cur_rgbd.get_cols(s);

    Eigen::MatrixXf I_xy(1,2);
    Eigen::MatrixXf Jp(2,6);

    Eigen::MatrixXf Jc(2,3);
    Eigen::MatrixXf Jq(3,6);

    // Eigen::MatrixXf J = Eigen::MatrixXf::Zero(w*h, 6);
    // Eigen::VectorXf residuals = Eigen::VectorXf::Zero(w*h);
    Eigen::Matrix3f K = cur_rgbd.getK(s);
    float fx = K(0,0);
    float fy = K(1,1);

    cv::Mat gray1 = cur_rgbd.get_gray(s);
    cv::Mat gray2 = prev_rgbd.get_gray(s);

    int iters = 20;
    float damping_factor = 1.0;

    float prev_err = std::numeric_limits<float>::max();

    int patch_size = 1;
        
    float ttl_err = 0.0;
    float cnt = 0;

    int pad = patch_size;
    //for(int i = 0; i < match_pixels.size(); i++){
    for(int x = pad; x < w-pad; x++){
        for(int y = pad; y < h-pad; y++){
        
            //int x = match_pixels[i].first;
            //int y = match_pixels[i].second;

            Eigen::Vector3f P1 = cur_rgbd.get_xyz(x, y, s);

            if(P1[2] > 3.5 || P1[2] < 0.1) continue;
            
            Eigen::Vector3f P1_ = T1 * P1;
            Eigen::Vector3f P1_reproj = T2.inverse() * P1_;

            float X = P1_reproj[0];
            float Y = P1_reproj[1];
            float Z = P1_reproj[2];

            if (Z < 0) continue;

            //Jp << fx*1/Z, 0, -fx*X/(Z*Z), -fx*(X*Y)/(Z*Z), fx*(1 + (X*X)/(Z*Z)), -fx*Y/Z,
            //        0, fy*1/Z, -fy*Y/(Z*Z), -fy*(1+(Y*Y)/(Z*Z)), fy*X*Y/(Z*Z), fy*X/Z;
            Jc << fx*1/Z, 0, -fx*Z/(Z*Z),0,fy*1/Z,-fy*Y/(Z*Z);
            Jq.block(0,0,3,3) = Eigen::Matrix3f::Identity();
            Jq.block(0,3,3,3) = -Sophus::SO3f::hat(P1_);
            Jq = T2.inverse().matrix().block(0,0,3,3) * Jq;
            // cout<<"Jq1\n";
            // cout<<Jq<<"\n";

            // numerical_jacobian(T1, T2, P1, Jq);
            // cout<<"Jq2\n";
            // cout<<Jq<<"\n";

            Jp = Jc * Jq;

            Eigen::Vector3f warped_points = cam2pixel(P1_reproj/Z, K);

            for(int x_offset = -(patch_size/2); x_offset <= patch_size/2; x_offset++){
                for(int y_offset = -(patch_size/2); y_offset <= patch_size/2; y_offset++){


                    if (warped_points[0]+(float)x_offset >= pad && warped_points[0]+(float)x_offset < w-pad && 
                        warped_points[1]+(float)y_offset >= pad && warped_points[1]+(float)y_offset < h-pad){
                        
                        I_xy(0,0) = 0.5 * (interpolate(gray2, warped_points[0]+(float)x_offset+1, warped_points[1]+(float)y_offset) - 
                                            interpolate(gray2, warped_points[0]+(float)x_offset-1, warped_points[1]+(float)y_offset));
                        I_xy(0,1) = 0.5 * (interpolate(gray2, warped_points[0]+(float)x_offset, warped_points[1]+(float)y_offset+1) - 
                                            interpolate(gray2, warped_points[0]+(float)x_offset, warped_points[1]+(float)y_offset-1));

                        int y2 = (int)warped_points[1];
                        int x2 = (int)warped_points[0];

                        Eigen::Vector3f P2 = prev_rgbd.get_xyz(x2, y2, s);
                        if((P1 - P2).norm() < 0.5){

                            float intensity = interpolate(gray2, 
                                                warped_points[0]+(float)x_offset,
                                                warped_points[1]+(float)y_offset);
                            float residual = (float)gray1.at<uint8_t>(y+y_offset, x+x_offset) - intensity;

                            
                            Eigen::MatrixXf J = -(I_xy * Jp);
                            H += J.transpose() * J;
                            b += -J.transpose() * residual;

                            ttl_err += std::abs(residual);
                            cnt++;
                        }
                    }
                }
            }
        }
    }
    cout<<"Dense error: "<<ttl_err/cnt<<"\n";
    return ttl_err/cnt;
}

float Slam::accu_jacobian_dense_from_world(RGBD& cur_rgbd, Sophus::SE3f& T, Eigen::MatrixXf& H, Eigen::VectorXf& b){
    Eigen::MatrixXf Jp(2,6);

    Eigen::MatrixXf Jc(2,3);
    Eigen::MatrixXf Jq(3,6);

    // Eigen::MatrixXf J = Eigen::MatrixXf::Zero(w*h, 6);
    // Eigen::VectorXf residuals = Eigen::VectorXf::Zero(w*h);
    int s = 1;

    int h = cur_rgbd.get_rows(s);
    int w = cur_rgbd.get_cols(s);
    Eigen::Matrix3f K = cur_rgbd.getK(s);
    float fx = K(0,0);
    float fy = K(1,1);

    cv::Mat cur_img = cur_rgbd.get_img(s);
    cv::Mat cur_depth = cur_rgbd.get_depth(s);

    int pad = 2;
    float ttl_err = 0.0;
    int cnt = 0;

    for(int i = 0; i < world_pnts.size(); i++){
        Eigen::Vector3f pnt = world_pnts[i];
        Eigen::Vector3f rgb = world_rgbs[i];
        pnt = T * pnt;

        float X = pnt[0];
        float Y = pnt[1];
        float Z = pnt[2];

        //Jp << fx*1/Z, 0, -fx*X/(Z*Z), -fx*(X*Y)/(Z*Z), fx*(1 + (X*X)/(Z*Z)), -fx*Y/Z,
        //        0, fy*1/Z, -fy*Y/(Z*Z), -fy*(1+(Y*Y)/(Z*Z)), fy*X*Y/(Z*Z), fy*X/Z;
        Jc << fx*1/Z, 0, -fx*Z/(Z*Z),0,fy*1/Z,-fy*Y/(Z*Z);  
        Jq.block(0,0,3,3) = Eigen::Matrix3f::Identity();
        Jq.block(0,3,3,3) = -Sophus::SO3f::hat(pnt);

        Jp = Jc * Jq; //2x6 = 2x3 * 3x6

        Eigen::Vector3f warped_points = cam2pixel(pnt/Z, K);
        if (warped_points[0] >= pad && warped_points[0] < w-pad && 
            warped_points[1] >= pad && warped_points[1] < h-pad){

            Eigen::MatrixXf J_rgb_p = Eigen::MatrixXf::Zero(3, 2);            
            Eigen::Vector3f warped_rgb; 
            Eigen::MatrixXf J_d_p = Eigen::MatrixXf::Zero(1, 2);
            float warped_d = 0.0;

            interpolate_with_jacobian(cur_img, warped_points[0], warped_points[1], warped_rgb, J_rgb_p);
            interpolate_with_jacobian(cur_depth, warped_points[0], warped_points[1], &warped_d, J_d_p);

            Eigen::VectorXf residual_rgb = rgb - warped_rgb;
            float residual_d = Z - warped_d/cur_rgbd.depth_factor;
            
            Eigen::MatrixXf J_rgbd_p = Eigen::MatrixXf::Zero(4, 2);
            J_rgbd_p.block(0,0,3,2) = J_rgb_p;
            J_rgbd_p.block(3,0,1,2) = J_d_p;

            Eigen::VectorXf residual = Eigen::VectorXf::Zero(4);
            residual.segment(0, 3) = residual_rgb;
            residual[3] = residual_d;

            Eigen::MatrixXf J = -J_rgbd_p * Jp;   //4x6 = 4x2 * 2x6
            J.block(3,0,1,6) += Jq.block(2,0,1,6);


            
            H += J.transpose() * J;
            b += -J.transpose() * residual;

            ttl_err += residual.norm();
            cnt++;
        }
        H /= (cnt * cnt);
        b /= (cnt * cnt);
    }
    std::cout<<"Dense from world: "<<ttl_err/cnt<<"\n";
    return ttl_err/cnt;
}

float Slam::accu_jacobian_dense_from_kf(RGBD& cur_rgbd, std::vector<int>& selected_kf, Sophus::SE3f& T, Eigen::MatrixXf& H, Eigen::VectorXf& b, int s, std::vector<float>& dense_err){
    Eigen::MatrixXf Jp(2,6);

    Eigen::MatrixXf Jc(2,3);
    Eigen::MatrixXf Jq(3,6);

    // Eigen::MatrixXf J = Eigen::MatrixXf::Zero(w*h, 6);
    // Eigen::VectorXf residuals = Eigen::VectorXf::Zero(w*h);
    // int s = 1;

    int h = cur_rgbd.get_rows(s);
    int w = cur_rgbd.get_cols(s);
    Eigen::Matrix3f K = cur_rgbd.getK(s);
    float fx = K(0,0);
    float fy = K(1,1);

    cv::Mat cur_img = cur_rgbd.get_img(s);
    cv::Mat cur_depth = cur_rgbd.get_depth(s);

    int pad = 2;
    float ttl_err = 0.0;
    int cnt = 0;

    std::vector<float> xyz_errs;
    std::vector<int> xyz_cnts;

    std::vector<RGBD*> gn_rgbd_list;
    //for(int i = std::max((int)rgbd_list.size()-5, 0); i < rgbd_list.size(); i++){
    for(int i : selected_kf){
        gn_rgbd_list.push_back(&rgbd_list[i]);
    }
    if ((num_frames-1 % kf_interval) != 0){
        gn_rgbd_list.push_back(&tmp_rgbd_list[1]);
    }

    // for(int i = std::max((int)rgbd_list.size()-5, 0); i < rgbd_list.size(); i++){
    for(int i = 0; i < gn_rgbd_list.size(); i++){

        Eigen::MatrixXf H_i(6, 6);
        Eigen::VectorXf b_i(6);

        float xyz_err = 0.0;
        int xyz_cnt = 0;

        for(int x = pad; x < w-pad; x++){
            for(int y = pad; y < h-pad; y++){
            
                Eigen::Vector3f pnt = gn_rgbd_list[i]->get_xyz(x, y, s);
                Eigen::Vector3f rgb = gn_rgbd_list[i]->get_rgb(x, y, s);

                pnt = gn_rgbd_list[i]->get_pose() * pnt;  // Local To global 

                pnt = T * pnt;  //Global to Local

                float X = pnt[0];
                float Y = pnt[1];
                float Z = pnt[2];

                Jc << fx*1/Z, 0, -fx*Z/(Z*Z),0,fy*1/Z,-fy*Y/(Z*Z);  
                Jq.block(0,0,3,3) = Eigen::Matrix3f::Identity();
                Jq.block(0,3,3,3) = -Sophus::SO3f::hat(pnt);

                Jp = Jc * Jq; //2x6 = 2x3 * 3x6

                Eigen::Vector3f warped_points = cam2pixel(pnt/Z, K);
                if (warped_points[0] >= pad && warped_points[0] < w-pad && 
                    warped_points[1] >= pad && warped_points[1] < h-pad){

                    Eigen::MatrixXf J_rgb_p = Eigen::MatrixXf::Zero(3, 2);            
                    Eigen::Vector3f warped_rgb; 
                    Eigen::MatrixXf J_d_p = Eigen::MatrixXf::Zero(1, 2);
                    float warped_d = 0.0;

                    interpolate_with_jacobian(cur_img, warped_points[0], warped_points[1], warped_rgb, J_rgb_p);
                    interpolate_with_jacobian(cur_depth, warped_points[0], warped_points[1], &warped_d, J_d_p);

                    warped_d /= cur_rgbd.depth_factor;
                    J_d_p /= cur_rgbd.depth_factor;

                    Eigen::Vector3f warped_xyz = pixel2cam(warped_points, K) * warped_d;

                    Eigen::VectorXf residual_rgb = rgb - warped_rgb;
                    Eigen::Vector3f residual_xyz = pnt - warped_xyz;

                    // residual_rgb /= 255.0;
                    // J_rgb_p /= 255.0;
                    xyz_err += residual_xyz.norm();
                    xyz_cnt++;

                    if(residual_xyz.norm() > 0.1 || residual_rgb.norm() > 30) continue;
                    // cout<<Z<<" "<<warped_d<<" "<<target_pnt[2]<<"\n";
                    // if(std::abs(residual_d) > 0.1){
                    //     cout<<residual_d<<"\n";
                    //     continue;
                    // }
                    
                    Eigen::MatrixXf J_rgbd_p = Eigen::MatrixXf::Zero(4, 2);
                    J_rgbd_p.block(0,0,3,2) = J_rgb_p;
                    // J_rgbd_p.block(3,0,1,2) = J_d_p;

                    // Eigen::MatrixXf J_rgbxyz_p = Eigen::MatrixXf::Zero(6, 2);

                    // Eigen::VectorXf residual = Eigen::VectorXf::Zero(6);
                    // residual.segment(0, 3) = residual_rgb * 0.2;
                    // residual.segment(3, 3) = residual_xyz;
                    // // residual[3] = residual_d;
                    // Eigen::MatrixXf J(6, 6);

                    // // Eigen::MatrixXf J = -J_rgbd_p * Jp;   //4x6 = 4x2 * 2x6
                    // J.block(0,0,3,6) = -J_rgb_p * Jp * 0.2;  //3x6 = 3x2 * 2x6
                    // J.block(3,0,3,6) = Jq - warped_points * J_d_p * Jp;

                    Eigen::MatrixXf J(2, 6);

                    J.block(0,0,1,6) = 0.1 * 2.0 * residual_rgb.transpose() * -J_rgb_p * Jp;
                    J.block(1,0,1,6) = 2.0 * residual_xyz.transpose() * (Jq - warped_points * J_d_p * Jp);

                    Eigen::VectorXf error(2);
                    error[0] = std::pow(residual_rgb.norm(), 2) * 0.1;
                    error[1] = std::pow(residual_xyz.norm(), 2);
                    
                    H += J.transpose() * J;
                    b += -J.transpose() * error;

                    // ttl_err += residual.norm();
                    ttl_err += error[0] + error[1];
                    cnt++;
                }
            }
        }
        float overlap = (float)xyz_cnt/(float)(h*w);

        xyz_errs.push_back(xyz_err/xyz_cnt);
        if (overlap > 0.2){
            dense_err.push_back(xyz_err/xyz_cnt);
        }
        else dense_err.push_back(0.0);
    }

    std::cout<<"Dense from kfs: "<<ttl_err/cnt<<" "<<cnt<<"\n";
    
    return ttl_err/cnt;
}

void Slam::add_points(RGBD& rgbd){
    // RGBD rgbd = rgbd_list.back();
    VecVector6f xyzrgb = rgbd.get_pc(0);
    Sophus::SE3f cur_pose = rgbd.get_pose();
    // cout<<cur_pose.matrix()<<"\n";

    std::vector<int> rand_indices;

    for(int i = 0; i < xyzrgb.size(); i++){
        rand_indices.push_back(i);
    }
    std::random_shuffle(rand_indices.begin(), rand_indices.end());

    int target_size = cloud->points.size() + 3000;
    // if(world_pnts.empty()) target_size = 5000;
    // else target_size = world_pnts.size() + 1000;

    
    for(int i = 0; i < rand_indices.size(); i++){

        int rand_id = rand_indices[i];
        Eigen::Vector3f cur_pnt; 
        Eigen::Vector3f cur_rgb;


        cur_pnt[0] = xyzrgb[rand_id][0];
        cur_pnt[1] = xyzrgb[rand_id][1];
        cur_pnt[2] = xyzrgb[rand_id][2];

        cur_rgb[0] = xyzrgb[rand_id][3];
        cur_rgb[1] = xyzrgb[rand_id][4];
        cur_rgb[2] = xyzrgb[rand_id][5];

        std::vector<int> visible_kf;

        VecVector3f match_pnts;
        VecVector3f match_rgbs;

        //for(int kf_i = 0; kf_i < rgbd_list.size(); kf_i++){
        for(int kf_i : cur_selected_kfs){
            Sophus::SE3f kf_pose = rgbd_list[kf_i].get_pose();
            // if(i == 0){
            //     cout<<kf_i<<"\n";
            //     cout<<kf_pose.matrix()<<"\n";
            // }
            
            Eigen::Vector3f proj_pnt = kf_pose.inverse() * cur_pose * cur_pnt;
            proj_pnt /= proj_pnt[2];
            proj_pnt = cam2pixel(proj_pnt, rgbd_list[kf_i].getK(0));

            Eigen::Vector3f proj_rgb = rgbd_list[kf_i].get_rgb(proj_pnt[0], proj_pnt[1], 0);
            proj_pnt = rgbd_list[kf_i].get_xyz(proj_pnt[0], proj_pnt[1], 0);

            // cout<<"here\n";
            // cout<<cur_pnt<<"\n";
            // cout<<proj_pnt<<"\n";
            // cout<<cur_rgb<<"\n";
            // cout<<proj_rgb<<"\n";

            if(proj_pnt[2] > 0 && (cur_pose * cur_pnt - kf_pose * proj_pnt).norm() < 0.1 && (cur_rgb - proj_rgb).norm() < 30){
                visible_kf.push_back(kf_i);
                // match_pnts.push_back(kf_pose * proj_pnt);
                // match_rgbs.push_back(proj_rgb);
            }

        }
        if(visible_kf.size() > 0 || cur_selected_kfs.empty()){
            world_pnts.push_back(cur_pose * cur_pnt);
            world_rgbs.push_back(cur_rgb);

            cur_pnt = cur_pose * cur_pnt;

            pcl::PointXYZRGB pnt_clr(cur_pnt[0],cur_pnt[1],cur_pnt[2],
                                     cur_rgb[0],cur_rgb[1],cur_rgb[2]);
            cloud->points.push_back(pnt_clr);
            
            //for(int j = 0; j < match_pnts.size(); j++){
            //    world_pnts.push_back(match_pnts[j]);
            //    world_rgbs.push_back(match_rgbs[j]);
            //}

            
        }
        if(cloud->points.size() == target_size) break; 
        //if(world_pnts.size() == target_size) break;
    }
    // cout<<wor<<"----------------------\n";
    

}


Eigen::VectorXf Slam::GN_solve(Eigen::VectorXf H_ll,
                    Eigen::MatrixXf H_pl,
                    Eigen::MatrixXf H_pp,
                    Eigen::VectorXf b_l,
                    Eigen::VectorXf b_p,
                    float damping_factor){

    Eigen::VectorXf damping_p = Eigen::VectorXf::Ones(H_ll.size());
    H_ll += damping_factor * damping_p;
    Eigen::MatrixXf damping_l = Eigen::MatrixXf::Identity(H_pp.rows(), H_pp.cols());
    H_pp += damping_factor * damping_l;
    
    Eigen::VectorXf H_ll_inv = diag_inverse(H_ll);


    Eigen::MatrixXf H = H_pp - matmul_mat_diag(H_pl, H_ll_inv) * H_pl.transpose();
    Eigen::VectorXf b = b_p - matmul_mat_diag(H_pl, H_ll_inv) * b_l;


    Eigen::VectorXf dx = Eigen::VectorXf::Zero(b_p.size()+b_l.size());

    // dx_p = H.inverse()*b;
    Eigen::VectorXf dx_p = H.inverse()*b;
    dx.segment(0, b_p.size()) = dx_p;
    // cout<<"here\n";
    // cout<<dx_p<<"\n";

    // b = -b_l - H_pl.inverse()*dx_p;
    // dx_l = H_ll_inv * (-b_l - H_pl.transpose()*dx_p);
    dx.segment(b_p.size(), b_l.size()) = matmul_diag_vec(H_ll_inv, (b_l - H_pl.transpose()*dx_p));
    return dx;
     
}

void Slam::BA(int iters){
    int rows = 0;

    //Construct observation graph
    std::vector<std::vector<int>> BA_graph;
    std::vector<VecVector3f> BA_measure;
    for(int i = 0; i < world_pnts.size(); i++){
        Eigen::Vector3f cur_pnt = world_pnts[i];
        Eigen::Vector3f cur_rgb = world_rgbs[i];

        // int visible = 0;
        std::vector<int> visible_kf;
        VecVector3f measures;
        
        for(int kf_i = 0; kf_i < rgbd_list.size(); kf_i++){
            Sophus::SE3f kf_pose = rgbd_list[kf_i].get_pose();
            Eigen::Vector3f proj_pnt = kf_pose.inverse() * cur_pnt;
            proj_pnt /= proj_pnt[2];
            proj_pnt = cam2pixel(proj_pnt, rgbd_list[kf_i].getK(0));

            Eigen::Vector3f proj_rgb = rgbd_list[kf_i].get_rgb(proj_pnt[0], proj_pnt[1], 0);
            proj_pnt = rgbd_list[kf_i].get_xyz(proj_pnt[0], proj_pnt[1], 0);

            if(proj_pnt[2] > 0 && (cur_pnt - kf_pose * proj_pnt).norm() < 0.05 && (cur_rgb - proj_rgb).norm() < 30){
                visible_kf.push_back(kf_i);
                measures.push_back(proj_pnt);
            }
        }
        BA_graph.push_back(visible_kf);
        BA_measure.push_back(measures);
        rows += visible_kf.size() * 3;
    }

    int cols = world_pnts.size()*3 + rgbd_list.size()*6;

    // Eigen::MatrixXf J = Eigen::MatrixXf::Zero(rows, cols);

    int kf_num = rgbd_list.size();
    int lm_vars = world_pnts.size()*3;
    int pose_vars = kf_num*6;
    
    float damping_factor = 1;

    float prev_err = std::numeric_limits<float>::max();


    for(int iter = 0; iter < iters; iter++){

        float ttl_err = 0.0;
        
        
        // Eigen::MatrixXf J = Eigen::MatrixXf::Zero(rows, cols);
        // Eigen::VectorXf Residual = Eigen::VectorXf::Zero(rows);

        // Eigen::MatrixXf H_ll = Eigen::MatrixXf::Zero(world_pnts.size()*3, world_pnts.size()*3);
        // Eigen::MatrixXf H_ll = Eigen::MatrixXf::Zero(world_pnts.size()*3, 3);
        // Eigen::DiagonalMatrix H_ll = Eigen::DiagonalMatrix::setZero(world_pnts.size()*3);
        Eigen::VectorXf H_ll = Eigen::VectorXf::Zero(lm_vars); //This is actually a diagonal matrix
        Eigen::MatrixXf H_pl = Eigen::MatrixXf::Zero(pose_vars, lm_vars);
        Eigen::MatrixXf H_pp = Eigen::MatrixXf::Zero(pose_vars, pose_vars);
        // Eigen::DiagonalMatrix H_pp = Eigen::DiagonalMatrix::setZero(kf_list.size()*6);

        Eigen::VectorXf b_l = Eigen::VectorXf::Zero(lm_vars);
        Eigen::VectorXf b_p = Eigen::VectorXf::Zero(pose_vars);

        int row = 0;
        for(int i = 0; i < world_pnts.size(); i++){
            

            Eigen::Vector3f cur_pnt = world_pnts[i];

            int observe_num_kf = BA_graph[i].size();

            //Jacobian between error and landmark i
            Eigen::MatrixXf J_l = Eigen::MatrixXf::Zero(observe_num_kf*3, 3);
            //Jacobian between error and keyframe poses
            Eigen::MatrixXf J_p = Eigen::MatrixXf::Zero(observe_num_kf*3, pose_vars);

            //Residual between landmark i and observations at keyframes
            Eigen::VectorXf residual = Eigen::VectorXf::Zero(observe_num_kf*3);

            

            for(int j = 0; j < observe_num_kf; j++){

                int kf_i = BA_graph[i][j];

                Sophus::SE3f kf_pose = rgbd_list[kf_i].get_pose();
                // Eigen::Vector3f proj_pnt = kf_pose.inverse() * cur_pnt;
                // proj_pnt /= proj_pnt[2];
                // proj_pnt = cam2pixel(proj_pnt, rgbd_list[kf_i].getK(0));

                // proj_pnt = rgbd_list[kf_i].get_xyz(proj_pnt[0], proj_pnt[1], 0);
                Eigen::Vector3f proj_pnt = BA_measure[i][j];
                proj_pnt = kf_pose * proj_pnt;

                Eigen::Vector3f e = cur_pnt - proj_pnt;

                J_l.block<3,3>(j*3, 0) = Eigen::Matrix3f::Identity();

                J_p.block<3,3>(j*3, kf_i * 6) = -Eigen::Matrix3f::Identity();
                J_p.block<3,3>(j*3, kf_i * 6 + 3) = Sophus::SO3f::hat(proj_pnt);

                // J.block<3,3>(row*3, pose_vars + i*3) = Eigen::Matrix3f::Identity();

                Eigen::MatrixXf err = Eigen::Matrix3f::Zero(3,3);
                err.diagonal() = e;

                // J.block<3,3>(row*3, pose_vars + i*3) = Eigen::Matrix3f::Identity();

                Eigen::MatrixXf par_dir = Eigen::MatrixXf::Zero(3, 6);
                par_dir.block<3,3>(0,0) = -Eigen::Matrix3f::Identity();
                par_dir.block<3,3>(0,3) = Sophus::SO3f::hat(proj_pnt);

                // J.block<3,3>(row*3, kf_i/10 * 6) = -Eigen::Matrix3f::Identity();
                // J.block<3,3>(row*3, kf_i/10 * 6 + 3) = Sophus::SO3f::hat(proj_pnt);
                // cout<<(err*par_dir).rows()<<" "<<(err*par_dir).cols()<<"\n";
                // J.block<3,6>(row*3, kf_i/10 * 6) = par_dir;

               
                residual.segment(j*3, 3) = e;
                // Residual.segment(row*3, 3) = e;
                ttl_err += e.norm();
                
                row++;
            }
            

            H_ll.segment(i*3, 3) = (J_l.transpose() * J_l).diagonal();

            H_pp += J_p.transpose() * J_p;
          

            H_pl.block(0, i*3, pose_vars, 3) = J_p.transpose() * J_l;
         
            b_l.segment(i*3, 3) = -J_l.transpose() * residual;
            b_p += -J_p.transpose() * residual;
            // cout<<i<<"\n";
        }

        Eigen::VectorXf dx = GN_solve(H_ll, H_pl, H_pp, b_l, b_p, damping_factor);
       

        Eigen::VectorXf dx_p = dx.segment(0, pose_vars);
        Eigen::VectorXf dx_l = dx.segment(pose_vars, lm_vars);

        for(int i = 0; i < rgbd_list.size(); i++){
            Sophus::SE3f cur_dx = Sophus::SE3f::exp(dx_p.segment(i*6, 6));
            Sophus::SE3f prev_pose = rgbd_list[i].get_pose();
            rgbd_list[i].set_pose(cur_dx * prev_pose);
        }

        for(int i = 0; i < world_pnts.size(); i++){
            Eigen::Vector3f cur_dx = dx_l.segment(i*3, 3);
            world_pnts[i] += cur_dx;
        }
        
        std::cout<<iter<<" "<<ttl_err<<"\n";
        if (ttl_err < prev_err){
            damping_factor *= 0.1;
            
        }
        prev_err = ttl_err;
    }
}

void Slam::BA_dense(int iters, int s){
    int num_kfs = rgbd_list.size();
    Eigen::MatrixXf H(num_kfs*6, num_kfs*6);
    Eigen::VectorXf b(num_kfs*6);
    Eigen::VectorXf grad(num_kfs*6);

    int pad = 3;
    float prev_err = std::numeric_limits<float>::max();
    for(int iter = 0; iter < iters; iter++){
    
        float ttl_err = 0.0;
        int cnt = 0;
        for(int i = 0; i < rgbd_list.size(); i++){
            RGBD rgbd_i = rgbd_list[i];
            
            int h = rgbd_i.get_rows(s);
            int w = rgbd_i.get_cols(s);

            Sophus::SE3f Ti = rgbd_i.get_pose();

            for(int x = pad; x < w - pad; x++){
                for(int y = pad; y < h - pad; y++){

                    Eigen::Vector3f cur_xyz = rgbd_i.get_xyz(x, y, s);
                    Eigen::Vector3f cur_rgb = rgbd_i.get_rgb(x, y, s);

                    cur_xyz = Ti * cur_xyz;  //Local to Global

                    Eigen::MatrixXf J_p_Ti(3, 6);
                    J_p_Ti.block(0,0,3,3) = Eigen::Matrix3f::Identity();
                    J_p_Ti.block(0,3,3,3) = -Sophus::SO3f::hat(cur_xyz);

                    for(int j = 0; j < rgbd_list.size(); j++){
                        if(i == j) continue;
                        RGBD rgbd_j = rgbd_list[j];

                        cv::Mat img_j = rgbd_j.get_img(s);
                        cv::Mat depth_j = rgbd_j.get_depth(s);

                        Eigen::Matrix3f K = rgbd_j.getK(s);
                        float fx = K(0,0);
                        float fy = K(1,1);

                        Sophus::SE3f Tj = rgbd_j.get_pose();
                        Eigen::Vector3f proj_xyz = Tj.inverse() * cur_xyz;    //Global to local

                        float X = proj_xyz[0];
                        float Y = proj_xyz[1];
                        float Z = proj_xyz[2];

                        Eigen::MatrixXf J_p_Tj(3, 6);
                        J_p_Tj.block(0,0,3,3) = Eigen::Matrix3f::Identity();
                        J_p_Tj.block(0,3,3,3) = -Sophus::SO3f::hat(proj_xyz);
                        J_p_Tj *= -Tj.inverse().Adj();

                        

                        Eigen::MatrixXf J_u_p(2,3);
                        
                        J_u_p << fx*1/Z, 0, -fx*Z/(Z*Z),0,fy*1/Z,-fy*Y/(Z*Z);  
                        
                        J_p_Ti = Tj.inverse().matrix().block(0,0,3,3) * J_p_Ti;


                        Eigen::Vector3f warped_uv = cam2pixel(proj_xyz/proj_xyz[2], K);
                        Eigen::Vector3f warped_rgb;
                        float d = 0.0;
                        Eigen::MatrixXf J_rgb_u(3, 2);
                        Eigen::MatrixXf J_d_u(1, 2);

                    

                        if(warped_uv[0] < pad ||  warped_uv[0] > w - pad || warped_uv[1] < pad || warped_uv[1] > h - pad) continue;

                        interpolate_with_jacobian(img_j, warped_uv[0], warped_uv[1], warped_rgb, J_rgb_u);
                        interpolate_with_jacobian(depth_j, warped_uv[0], warped_uv[1], &d, J_d_u);

                        d /= rgbd_j.depth_factor;
                        J_d_u /= rgbd_j.depth_factor;

                        Eigen::VectorXf warped_xyz = pixel2cam(warped_uv, K);
                        warped_xyz *= d;

                        Eigen::Vector3f residual_rgb = cur_rgb - warped_rgb;
                        Eigen::Vector3f residual_xyz = proj_xyz - warped_xyz;

                        residual_rgb /= 255.0;
                        warped_rgb /= 255.0;

                        // std::cout<<cur_rgb.transpose()<<" "<<warped_rgb.transpose()<<"\n";
                        // std::cout<<proj_xyz.transpose()<<" "<<warped_xyz.transpose()<<"\n";

                        if(residual_rgb.norm() > 0.2 || residual_xyz.norm() > 0.2) continue;

                        Eigen::MatrixXf J_rgb_i(3, 6);
                        J_rgb_i = -J_rgb_u * J_u_p * J_p_Ti;

                        Eigen::MatrixXf J_xyz_i(3, 6);
                        J_xyz_i = J_p_Ti - warped_uv * J_d_u * J_u_p * J_p_Ti;

                        Eigen::MatrixXf J_rgb_j(3, 6);
                        J_rgb_j = -J_rgb_u * J_u_p * J_p_Tj;

                        Eigen::MatrixXf J_xyz_j(3, 6);
                        J_xyz_j = J_p_Tj - warped_uv * J_d_u * J_u_p * J_p_Tj;


                        Eigen::MatrixXf Ji(2, 6);

                        Ji.block(0,0,1,6) = 2.0 * residual_rgb.transpose() * J_rgb_i;
                        Ji.block(1,0,1,6) = 2.0 * residual_xyz.transpose() * J_xyz_i;

                        Eigen::MatrixXf Jj(2, 6);

                        Jj.block(0,0,1,6) = 2.0 * residual_rgb.transpose() * J_rgb_j;
                        Jj.block(1,0,1,6) = 2.0 * residual_xyz.transpose() * J_xyz_j;

                        Eigen::VectorXf error(2);
                        error[0] = std::pow(residual_rgb.norm(), 2);
                        error[1] = std::pow(residual_xyz.norm(), 2);

                        H.block(i*6, i*6, 6, 6) += Ji.transpose() * Ji;
                        H.block(j*6, j*6, 6, 6) += Jj.transpose() * Jj;
                        
                        H.block(i*6, j*6, 6, 6) += Ji.transpose() * Jj;
                        b.segment(i*6, 6) += -Ji.transpose() * error;

                        H.block(j*6, i*6, 6, 6) += Jj.transpose() * Ji;
                        b.segment(j*6, 6) += -Jj.transpose() * error;

                        grad.segment(i*6, 6) += Ji;
                        grad.segment(j*6, 6) += Jj;

                        // H += J.transpose() * J;
                        // b += -J.transpose() * error;

                        // ttl_err += residual.norm();
                        ttl_err += error[0] + error[1];
                        cnt++;

                    }
                }
            }
   
        }
        std::cout<<"BA dense "<<ttl_err/cnt<<" "<<cnt<<"\n";

        Eigen::MatrixXf damping = Eigen::MatrixXf::Identity(num_kfs*6, num_kfs*6) * 0.1;

        // H += damping;

        Eigen::VectorXf dx = H.ldlt().solve(b);

        for(int i = 0; i < dx.size(); i++){
            if (dx[i] > 0.1) dx[i] = 0.1;
            if (dx[i] < -0.1) dx[i] = -0.1;
            if (isnan(dx[i])){
                dx[i] = 0.0;
                // std::cout<<H<<"\n";
                // std::cout<<"\n";
                // std::cout<<b.transpose()<<"\n";
            } 
        }

        // std::cout<<dx.transpose()<<"\n";
        for(int i = 0; i < rgbd_list.size(); i++){
            Sophus::SE3f prev_pose = rgbd_list[i].get_pose();
            Sophus::SE3f update_pose = Sophus::SE3f::exp(dx.segment(i*6, 6)) * prev_pose;
            rgbd_list[i].set_pose(update_pose);
        }

        if (ttl_err < prev_err){
            damping *= 10;
        }
        else damping *= 0.1;

        prev_err = ttl_err;

        float alpha = 1e-9;
        
    }
}

void Slam::BA_sparse(int iters){
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();

    VecVector3f match_pnts1;
    VecVector3f match_pnts2;
    std::vector<std::pair<int, int>> match_ids;
    for(int i = 0; i < rgbd_list.size(); i++){
        for(int j = i+1; j < rgbd_list.size(); j++){
            // if(i == j) continue;


            std::vector<cv::KeyPoint> keypoints_i = rgbd_list[i].get_keypoints(); 
            cv::Mat descriptors_i = rgbd_list[i].get_descriptors();

            std::vector<cv::KeyPoint> keypoints_j = rgbd_list[j].get_keypoints(); 
            cv::Mat descriptors_j = rgbd_list[j].get_descriptors();


            std::vector<std::vector<cv::DMatch>> knn_match;
            matcher->knnMatch(descriptors_i, descriptors_j, knn_match, 2);
                        
            VecVector3f pnts_i;
            VecVector3f pnts_j;

            std::vector<int> des_indexs1; //no use
            std::vector<int> des_indexs2; //no use

            std::vector<cv::KeyPoint> filter_keypoints_i;
            std::vector<cv::KeyPoint> filter_keypoints_j;

            std::vector<cv::DMatch> good_match;
            filter_matches(knn_match, 
                        good_match,
                        keypoints_i, keypoints_j, 
                        filter_keypoints_i, filter_keypoints_j,
                        des_indexs1, des_indexs2,
                        pnts_i, pnts_j, 0.6);

            // std::cout<<i<<" "<<j<<" "<<pnts_i.size()<<"\n";
            if(pnts_i.size() < 50) continue;
            // VecVector3f init_pnts1;
            

            //Transform points to 3D
            for (int k = 0; k < pnts_i.size(); k++){
                int x1 = (int)pnts_i[k][0];
                int y1 = (int)pnts_i[k][1];
                int x2 = (int)pnts_j[k][0];
                int y2 = (int)pnts_j[k][1];
                Eigen::Vector3f xyz1 = rgbd_list[i].get_xyz(x1, y1, 0);
                Eigen::Vector3f xyz2 = rgbd_list[j].get_xyz(x2, y2, 0);

                // init_pnts1.push_back(xyz1);
                // init_pnts2.push_back(xyz2);
                // local_match_pnts1.push_back(xyz2);
                // local_match_pnts2.push_back(xyz1);
                match_pnts1.push_back(xyz1);
                match_pnts2.push_back(xyz2);
                match_ids.push_back({i,j});
            }
        }
    }
    int kf_nums = rgbd_list.size();
    float prev_err = std::numeric_limits<float>::max();
    float damping_factor = 1.0;
    for(int iter = 0; iter < iters; iter++){
        Eigen::MatrixXf H(kf_nums * 6, kf_nums * 6);
        Eigen::VectorXf b(kf_nums * 6);
        

        float err = 0.0;
        for(int i = 0; i < match_pnts1.size(); i++){
            Eigen::Vector3f pnt1 = match_pnts1[i];
            Eigen::Vector3f pnt2 = match_pnts2[i];
            int kf_i = match_ids[i].first;
            int kf_j = match_ids[i].second;
            Sophus::SE3f Ti = rgbd_list[kf_i].get_pose();
            Sophus::SE3f Tj = rgbd_list[kf_j].get_pose();
            pnt1 = Ti * pnt1;
            pnt2 = Tj * pnt2;

            Eigen::Vector3f residual = pnt1 - pnt2;

            // if(residual.norm() > 0.5) {
            //     std::cout<<residual.norm()<<"\n";
            //     continue;
            // }
            Eigen::MatrixXf J_i(3, 6);
            Eigen::MatrixXf J_j(3, 6);
            J_i.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();
            J_i.block(0, 3, 3, 3) = -Sophus::SO3f::hat(pnt1);

            J_j.block(0, 0, 3, 3) = -Eigen::Matrix3f::Identity();
            J_j.block(0, 3, 3, 3) = Sophus::SO3f::hat(pnt2);

            H.block(kf_i*6, kf_i*6, 6, 6) += J_i.transpose() * residual * residual.transpose() * J_i;
            H.block(kf_j*6, kf_j*6, 6, 6) += J_j.transpose() * residual * residual.transpose() * J_j;
            H.block(kf_i*6, kf_j*6, 6, 6) += J_i.transpose() * residual * residual.transpose() * J_j;
            H.block(kf_j*6, kf_i*6, 6, 6) += J_j.transpose() * residual * residual.transpose() * J_i;

            b.segment(kf_i*6, 6) += -J_i.transpose() * residual * residual.norm();
            b.segment(kf_j*6, 6) += -J_j.transpose() * residual * residual.norm(); 

            err += residual.norm();
            
        }
        std::cout<<"BA sparse: "<<err/match_pnts1.size()<<"\n";

        H /= (match_pnts1.size() * match_pnts1.size());
        b /= (match_pnts1.size() * match_pnts1.size());

        // std::cout<<H<<"\n";
        // std::cout<<"\n";
        // std::cout<<b.transpose()<<"\n";
        
        Eigen::MatrixXf damping = Eigen::MatrixXf::Identity(6, 6) ;
        H += damping * damping_factor;
        Eigen::VectorXf dx = H.ldlt().solve(b);
        for(int i = 0; i < dx.size(); i++){
            if(isnan(dx[i]) || std::abs(dx[i]) > 0.1) dx[i] = 0;
        }
        for(int i = 0; i < rgbd_list.size(); i++){
            Sophus::SE3f prev_pose = rgbd_list[i].get_pose();
            // std::cout<<dx.segment(i*6, 6).transpose()<<"\n";
            Sophus::SE3f update_pose = Sophus::SE3f::exp(dx.segment(i*6, 6)) * prev_pose;
            rgbd_list[i].set_pose(update_pose);
        }

        if (err < prev_err){
            damping_factor *= 0.1;
        }
        else{
            damping_factor *= 10;
        }
        prev_err = err;
    }
    
}


void Slam::visualize_all(){

    // std::vector<RGBD> cur_list = rgbd_list;
    //for(std::vector<RGBD> cur_list: segments){

	    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

        for(int i = 0; i < rgbd_list.size(); i++){
        //for(int i: keyframe_idxs){
            // if(i % 10 != 0) continue;
            VecVector6f pc = rgbd_list[i].get_pc(1);
            for(Vector6f p: pc){
                Eigen::Vector3f coord = p.head(3);
                // if(coord[2] > 3.0) continue;
                // Sophus::SE3f tmp = rgbd_list[i].get_pose();
                // std::cout<<tmp.matrix()<<"\n";
                coord = rgbd_list[i].get_pose() * coord;
                pcl::PointXYZRGB pnt_clr(coord[0],coord[1],coord[2],p[3],p[4],p[5]);
                point_cloud_ptr->points.push_back(pnt_clr);

            }
        }

        // pcl::io::savePLYFileBinary(dataset_name+"/test_pcd.ply", *point_cloud_ptr);
        pcl::visualization::CloudViewer viewer("Cloud Viewer");
	    viewer.showCloud (point_cloud_ptr);
	    while (!viewer.wasStopped())
  	    {
  	    }
    //}
}


void Slam::visualize_kf(){

    // std::vector<RGBD> cur_list = rgbd_list;
    //for(std::vector<RGBD> cur_list: segments){

	    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

        for(int i = 0; i < rgbd_list.size(); i++){
        //for(int i: keyframe_idxs){
            // if(i % 10 != 0) continue;
            VecVector6f pc = rgbd_list[i].get_pc(1);
            for(Vector6f p: pc){
                Eigen::Vector3f coord = p.head(3);
                // if(coord[2] > 3.0) continue;
                // Sophus::SE3f tmp = rgbd_list[i].get_pose();
                // std::cout<<tmp.matrix()<<"\n";
                coord = rgbd_list[i].get_pose() * coord;
                pcl::PointXYZRGB pnt_clr(coord[0],coord[1],coord[2],p[3],p[4],p[5]);
                point_cloud_ptr->points.push_back(pnt_clr);

            }
        }

     

        // pcl::io::savePLYFileBinary(dataset_name+"/test_pcd.ply", *point_cloud_ptr);
        pcl::visualization::CloudViewer viewer("Cloud Viewer");
	    viewer.showCloud (point_cloud_ptr);
	    while (!viewer.wasStopped())
  	    {
  	    }
    //}
}


void Slam::visualize_wp(){

    // std::vector<RGBD> cur_list = rgbd_list;
    //for(std::vector<RGBD> cur_list: segments){

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

    cout<<world_pnts.size()<<"\n";
    for(int i = 0; i < world_pnts.size(); i++){
    //for(int i: keyframe_idxs){
        
        Eigen::Vector3f coord = world_pnts[i];
        Eigen::Vector3f color = world_rgbs[i];
        // if(coord[2] > 3.0) continue;
        // Sophus::SE3f tmp = rgbd_list[i].get_pose();
        // std::cout<<tmp.matrix()<<"\n";
        // coord = rgbd_list[i].get_pose() * coord;
        pcl::PointXYZRGB pnt_clr(coord[0],coord[1],coord[2],color[0],color[1],color[2]);
        point_cloud_ptr->points.push_back(pnt_clr);

        
    }

    // pcl::io::savePLYFileBinary(dataset_name+"/test_pcd.ply", *point_cloud_ptr);
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud (point_cloud_ptr);
    while (!viewer.wasStopped())
    {
    }
    //}
}
