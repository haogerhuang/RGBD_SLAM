#include <Eigen/Core>
#include <Eigen/LU>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "../include/utils.h"

Eigen::Vector3f pixel2cam(const Eigen::Vector3f &p, const Eigen::Matrix3f &K){
    Eigen::Vector3f camera_pnt = K.inverse() * p;
    //Eigen::Vector3d camera_pnt = K * p;
    camera_pnt = camera_pnt/camera_pnt[2];
    return camera_pnt;
}

Eigen::Vector3f cam2pixel(const Eigen::Vector3f &p, const Eigen::Matrix3f &K){
    Eigen::Vector3f pixel_pnt = K * p;
    pixel_pnt = pixel_pnt/pixel_pnt[2];
    return pixel_pnt;
}

Eigen::MatrixXf matmul_mat_diag(const Eigen::MatrixXf &mat1, const Eigen::VectorXf& mat2){
    int rows = mat1.rows();
    int cols = mat1.cols();
    Eigen::MatrixXf res = Eigen::MatrixXf::Zero(rows, cols); 
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            res(i,j) = mat1(i,j) * mat2[j];
        }
    }
    return res;
}
Eigen::VectorXf matmul_diag_vec(const Eigen::VectorXf& mat1, const Eigen::VectorXf& mat2){
    Eigen::VectorXf res = Eigen::VectorXf::Zero(mat1.size());
    for(int i = 0; i < mat1.size(); i++){
        res[i] = mat1[i] * mat2[i];
    }
    return res;
}



bool check_identical(const Eigen::MatrixXf& mat1, const Eigen::MatrixXf& mat2){
    for(int i = 0; i < mat1.rows(); i++){
        for(int j = 0; j < mat1.cols(); j++){
            if(std::abs(mat1(i,j)-mat2(i,j)) > 1e-3) return false;
        }
    }
    return true;
}

Eigen::VectorXf diag_inverse(const Eigen::VectorXf &mat){
    Eigen::VectorXf res = Eigen::VectorXf::Zero(mat.size());
    for(int i = 0; i < mat.size(); i++){
        if(mat[i]==0) res[i] = mat[i];
        else res[i] = 1/mat[i];
    }
    return res;
}

void numerical_jacobian(const Sophus::SE3f& T1, const Sophus::SE3f& T2, const Eigen::Vector3f& pnt, Eigen::MatrixXf& Jacobian, float h){
    Eigen::Vector3f res_pnt = T2.inverse() * T1 * pnt;
    // Eigen::Vector3f res_pnt = T1 * pnt;
    
    Sophus::Vector6f se3 = T1.log();
    Sophus::Vector6f se3_perturbed = se3;
    
    for(int i = 0; i < 6; i++){
        
        se3_perturbed[i] += h;
        // std::cout<<"per\n";
        // std::cout<<se3_perturbed.matrix()<<"\n";
        Sophus::SE3f T1_perturbed = Sophus::SE3f::exp(se3_perturbed);
        // std::cout<<"t\n";
        // std::cout<<T1_perturbed.matrix()<<"\n";
        Eigen::Vector3f res_pnt_h = T2.inverse() * T1_perturbed * pnt;
        Jacobian.col(i) = (res_pnt_h - pnt)/h;
        se3_perturbed[i] = se3[i];
    }
}

Sophus::SE3f kabsh_algorithm(
    const VecVector3f& pnts1_3d,
    const VecVector3f& pnts2_3d
){
    Eigen::Vector3f cen1(0,0,0); 
    Eigen::Vector3f cen2(0,0,0); 
    for (size_t i = 0; i < pnts1_3d.size(); i++){
        cen1 += pnts1_3d[i]; 
        cen2 += pnts2_3d[i]; 
    }
    cen1 = cen1 / float(pnts1_3d.size());
    cen2 = cen2 / float(pnts1_3d.size());

    Eigen::Vector3f t = cen2 - cen1;

    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();

    for (size_t i = 0; i < pnts1_3d.size(); i++){
        H(0,0) += pnts1_3d[i][0] * pnts2_3d[i][0];
        H(0,1) += pnts1_3d[i][0] * pnts2_3d[i][1];
        H(0,2) += pnts1_3d[i][0] * pnts2_3d[i][2];

        H(1,0) += pnts1_3d[i][1] * pnts2_3d[i][0];
        H(1,1) += pnts1_3d[i][1] * pnts2_3d[i][1];
        H(1,2) += pnts1_3d[i][1] * pnts2_3d[i][2];

        H(2,0) += pnts1_3d[i][2] * pnts2_3d[i][0];
        H(2,1) += pnts1_3d[i][2] * pnts2_3d[i][1];
        H(2,2) += pnts1_3d[i][2] * pnts2_3d[i][2];
    }

    Eigen::JacobiSVD<Eigen::Matrix3f > svd (H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f u = svd.matrixU();
    Eigen::Matrix3f v = svd.matrixV();
    
    Eigen::Matrix3f s = Eigen::Matrix3f::Identity();
    s(2,2) = (v*u).determinant();

    Eigen::Matrix3f R = v * s * u.transpose();

    Sophus::SE3f T(R, -R*cen1);

    Eigen::Matrix3f R1 = Eigen::Matrix3f::Identity();

    Sophus::SE3f T1(R1, -R1*cen2);

    return T1.inverse() * T;
}



float interpolate(const cv::Mat& img, float x, float y, bool print){
    int rows = img.rows;
    int cols = img.cols;

    if (x < 0 || x >= cols || y < 0 || y >= rows) return 0;

    int x0 = std::floor(x);
    int x1 = x0 + 1;
    int y0 = std::floor(y);
    int y1 = y0 + 1;
    
    float x0_w = x1 - x;
    float x1_w = 1 - x0_w;
    float y0_w = y1 - y;
    float y1_w = 1 - y0_w;

    if (x0 < 0 || x0 >= cols)
        x0_w = 0;
    if (x1 < 0 or x1 >= cols)
        x1_w = 0;
    if (y0 < 0 or y0 >= rows)
        y0_w = 0;
    if (y1 < 0 or y1 >= rows)
        y1_w = 0;

    float w00 = x0_w * y0_w;
    float w01 = x0_w * y1_w;
    float w10 = x1_w * y0_w;
    float w11 = x1_w * y1_w;

    int channels = img.channels();
    int step = img.step;
    float res = 0;
    if (channels > 1){
        if (w00 > 0){
            float intensity = (float)img.data[y0*step+x0*channels] +\
                              (float)img.data[y0*step+x0*channels+1] +\
                              (float)img.data[y0*step+x0*channels+2];
            intensity = intensity/3;
            res += intensity * w00;
        }
        if (w01 > 0){
            float intensity = (float)img.data[y1*step+x0*channels] +\
                              (float)img.data[y1*step+x0*channels+1] +\
                              (float)img.data[y1*step+x0*channels+2];
            intensity = intensity/3;
            res += intensity * w01;
        }
        if (w10 > 0){
            float intensity = (float)img.data[y0*step+x1*channels] +\
                              (float)img.data[y0*step+x1*channels+1] +\
                              (float)img.data[y0*step+x1*channels+2];
            intensity = intensity/3;
            res += intensity * w10;
        }
        if (w11 > 0){
            float intensity = (float)img.data[y1*step+x1*channels] +\
                              (float)img.data[y1*step+x1*channels+1] +\
                              (float)img.data[y1*step+x1*channels+2];
            intensity = intensity/3;
            res += intensity * w11;
        }
    }
    else{
        if (img.type() == 0){
            if (w00 > 0)
                res += (float)img.at<uint8_t>(y0, x0) * w00;
            if (w01 > 0)
                res += (float)img.at<uint8_t>(y1, x0) * w01;
            if (w10 > 0)
                res += (float)img.at<uint8_t>(y0, x1) * w10;
            if (w11 > 0)
                res += (float)img.at<uint8_t>(y1, x1) * w11;
        }
        else if(img.type() == 2){
            if (w00 > 0)
                res += (float)img.at<uint16_t>(y0, x0) * w00;
            if (w01 > 0)
                res += (float)img.at<uint16_t>(y1, x0) * w01;
            if (w10 > 0)
                res += (float)img.at<uint16_t>(y0, x1) * w10;
            if (w11 > 0)
                res += (float)img.at<uint16_t>(y1, x1) * w11;
        }
        else{
            if (w00 > 0)
                res += img.at<float>(y0, x0) * w00;
            if (w01 > 0)
                res += img.at<float>(y1, x0) * w01;
            if (w10 > 0)
                res += img.at<float>(y0, x1) * w10;
            if (w11 > 0)
                res += img.at<float>(y1, x1) * w11;
        }
    }
    
    //std::cout<<"x y"<<x<<" "<<y<<"\n";
    if(print) std::cout<<"inter "<<w00<<" "<<w01<<" "<<w10<<" "<<w11<<" "<<res<<"\n";

    // if ((w00 + w01 + w10 + w11) > 1e-10)
        // res = res / (w00+w01+w10+w11);
    // else res = 0;
    return res;

}

void interpolate_with_jacobian(const cv::Mat& img, float x, float y, Eigen::Vector3f& res, Eigen::MatrixXf& Jacobian){
    int rows = img.rows;
    int cols = img.cols;

    // Eigen::Vector3f res;

    if (x < 0 || x >= cols || y < 0 || y >= rows) return;

    int x0 = std::floor(x);
    int x1 = x0 + 1;
    int y0 = std::floor(y);
    int y1 = y0 + 1;
    
    float x0_w = x1 - x;
    float x1_w = 1 - x0_w;
    float y0_w = y1 - y;
    float y1_w = 1 - y0_w;

    if (x0 < 0 || x0 >= cols)
        x0_w = 0;
    if (x1 < 0 or x1 >= cols)
        x1_w = 0;
    if (y0 < 0 or y0 >= rows)
        y0_w = 0;
    if (y1 < 0 or y1 >= rows)
        y1_w = 0;

    float w00 = x0_w * y0_w;
    float w01 = x0_w * y1_w;
    float w10 = x1_w * y0_w;
    float w11 = x1_w * y1_w;

    Eigen::MatrixXf J_w_p = Eigen::MatrixXf::Zero(4, 2);
    
    J_w_p(0,0) = -y0_w; // dw00/dx = y0_w * d(x0_w)/dx = y0_w * -1 
    J_w_p(0,1) = -x0_w; // dw00/dy = x0_w * d(y0_w)/dy = x0_w * -1
    J_w_p(1,0) = -y1_w; // dw01/dx = y1_w * d(x0_w)/dx = y1_w * -1 
    J_w_p(1,1) = x0_w;  // dw01/dy = x0_w * d(y1_w)/dy = x0_w * 1
    J_w_p(2,0) = y0_w;  // dw10/dx = y0_w * d(x1_w)/dx = y0_w * 1 
    J_w_p(2,1) = -x1_w;  // dw10/dy = x1_w * d(y0_w)/dy = x1_w * -1
    J_w_p(3,0) = y1_w;  // dw11/dx = y1_w * d(x1_w)/dx = y1_w * 1 
    J_w_p(3,1) = x1_w;  // dw11/dy = x1_w * d(y1_w)/dy = x1_w * 1

    int channels = img.channels();
    int step = img.step;
    

    Eigen::MatrixXf J_I_w = Eigen::MatrixXf::Zero(3, 4);
    

    for(int c = 0; c < channels; c++){

        float res_c = 0;
        if (w00 > 0){
            float intensity = (float)img.data[y0*step+x0*channels+c];
                             
            res_c += intensity * w00;

            J_I_w(c,0) = intensity;
            
        }
        if (w01 > 0){
            float intensity = (float)img.data[y1*step+x0*channels+c];
                                          
            res_c += intensity * w01;

            J_I_w(c,1) = intensity;
            
        }
        if (w10 > 0){
            float intensity = (float)img.data[y0*step+x1*channels+c];
                              
            res_c += intensity * w10;

            J_I_w(c,2) = intensity;
            

        }
        if (w11 > 0){
            float intensity = (float)img.data[y1*step+x1*channels+c];
            
            res_c += intensity * w11;

            J_I_w(c,3) = intensity;
        }
        res[c] = res_c;
    }

    Eigen::MatrixXf tmp = J_I_w * J_w_p;

    for(int i = 0; i < channels; i++){
        for(int j = 0; j < 2; j++){
            Jacobian(i,j) = tmp(i,j);
        }
    }
    
    
    //std::cout<<"x y"<<x<<" "<<y<<"\n";
    //std::cout<<"inter "<<w00<<" "<<w01<<" "<<w10<<" "<<w11<<" "<<res<<"\n";

    // if ((w00 + w01 + w10 + w11) > 1e-10)
    //     res = res / (w00+w01+w10+w11);
    // else res = 0;

}

void interpolate_with_jacobian(const cv::Mat& img, float x, float y, float* res, Eigen::MatrixXf& Jacobian){
    int rows = img.rows;
    int cols = img.cols;

    // Eigen::Vector3f res;

    if (x < 0 || x >= cols || y < 0 || y >= rows) return;

    int x0 = std::floor(x);
    int x1 = x0 + 1;
    int y0 = std::floor(y);
    int y1 = y0 + 1;
    
    float x0_w = x1 - x;
    float x1_w = 1 - x0_w;
    float y0_w = y1 - y;
    float y1_w = 1 - y0_w;

    if (x0 < 0 || x0 >= cols)
        x0_w = 0;
    if (x1 < 0 or x1 >= cols)
        x1_w = 0;
    if (y0 < 0 or y0 >= rows)
        y0_w = 0;
    if (y1 < 0 or y1 >= rows)
        y1_w = 0;

    float w00 = x0_w * y0_w;
    float w01 = x0_w * y1_w;
    float w10 = x1_w * y0_w;
    float w11 = x1_w * y1_w;

    Eigen::MatrixXf J_w_p = Eigen::MatrixXf::Zero(4, 2);
    
    J_w_p(0,0) = -y0_w; // dw00/dx = y0_w * d(x0_w)/dx = y0_w * -1 
    J_w_p(0,1) = -x0_w; // dw00/dy = x0_w * d(y0_w)/dy = x0_w * -1
    J_w_p(1,0) = -y1_w; // dw01/dx = y1_w * d(x0_w)/dx = y1_w * -1 
    J_w_p(1,1) = x0_w;  // dw01/dy = x0_w * d(y1_w)/dy = x0_w * 1
    J_w_p(2,0) = y0_w;  // dw10/dx = y0_w * d(x1_w)/dx = y0_w * 1 
    J_w_p(2,1) = -x1_w;  // dw10/dy = x1_w * d(y0_w)/dy = x1_w * -1
    J_w_p(3,0) = y1_w;  // dw11/dx = y1_w * d(x1_w)/dx = y1_w * 1 
    J_w_p(3,1) = x1_w;  // dw11/dy = x1_w * d(y1_w)/dy = x1_w * 1

    int channels = img.channels();
    int step = img.step;
    

    Eigen::MatrixXf J_I_w = Eigen::MatrixXf::Zero(1, 4);
    
    

    float res_c = 0;
    if (w00 > 0){
        float intensity = (float)img.at<uint16_t>(y0, x0);

                            
        res_c += intensity * w00;

        J_I_w(0,0) = intensity;
        
    }
    if (w01 > 0){
        float intensity = (float)img.at<uint16_t>(y1, x0);
                                        
        res_c += intensity * w01;

        J_I_w(0,1) = intensity;
        
    }
    if (w10 > 0){
        float intensity = (float)img.at<uint16_t>(y0, x1);
                            
        res_c += intensity * w10;

        J_I_w(0,2) = intensity;
        

    }
    if (w11 > 0){
        float intensity = (float)img.at<uint16_t>(y1, x1);
        
        res_c += intensity * w11;

        J_I_w(0,3) = intensity;
    }
    *res = res_c;
    // std::cout<<res_c<<"\n";

    Eigen::MatrixXf tmp = J_I_w * J_w_p;

    // for(int i = 0; i < channels; i++){
    //     for(int j = 0; j < 2; j++){
    //         Jacobian(i,j) = tmp(i,j);
    //     }
    // }
    Jacobian.row(0) = tmp.row(0);
    
    
    //std::cout<<"x y"<<x<<" "<<y<<"\n";
    //std::cout<<"inter "<<w00<<" "<<w01<<" "<<w10<<" "<<w11<<" "<<res<<"\n";

    // if ((w00 + w01 + w10 + w11) > 1e-10)
    //     res = res / (w00+w01+w10+w11);
    // else res = 0;

}


void numerical_jacobian(const Sophus::SE3f& T, const Eigen::Vector3f& pnt, const Eigen::Matrix3f& K,
                        const Eigen::Vector3f& rgb, const cv::Mat& rgb_img, 
                        Eigen::MatrixXf& J_T, Eigen::MatrixXf& J_pnt, float h){

    Eigen::Vector3f proj_pnt = T * pnt;
    Eigen::Vector3f warped_points = cam2pixel(proj_pnt/proj_pnt[2], K);

    Eigen::MatrixXf tmp_J(3, 2);
    Eigen::Vector3f warped_rgb;
    interpolate_with_jacobian(rgb_img, warped_points[0], warped_points[1], warped_rgb, tmp_J);

    float ori_err = std::pow((rgb - warped_rgb).norm(), 2);

    Eigen::Vector3f tmp_pnt = pnt;
    for(int i = 0; i < 3; i++){
        tmp_pnt[i] += h;
        proj_pnt = T * tmp_pnt;
        warped_points = cam2pixel(proj_pnt/proj_pnt[2], K);
        
        
        interpolate_with_jacobian(rgb_img, warped_points[0], warped_points[1], warped_rgb, tmp_J);
        float res_err = std::pow((rgb - warped_rgb).norm(), 2);
        J_pnt(0, i) =  (res_err - ori_err)/h;
        tmp_pnt[i] = pnt[i];
    }

    Eigen::VectorXf se3 = T.log();
    Eigen::VectorXf se3_tmp = se3;

    for(int i = 0; i < 6; i++){
        se3_tmp[i] += h;

        proj_pnt = Sophus::SE3f::exp(se3_tmp) * pnt;
        warped_points = cam2pixel(proj_pnt/proj_pnt[2], K);
        
        interpolate_with_jacobian(rgb_img, warped_points[0], warped_points[1], warped_rgb, tmp_J);
        float res_err = std::pow((rgb - warped_rgb).norm(), 2);
        J_T(0, i) =  (res_err - ori_err)/h;
        se3_tmp[i] = se3[i];
    }


}

void numerical_jacobian(const Sophus::SE3f& T, const Eigen::Vector3f& pnt, const Eigen::Matrix3f& K,
                        const cv::Mat& depth_img, Eigen::MatrixXf& J_T, Eigen::MatrixXf& J_pnt, float h){

    Eigen::Vector3f proj_pnt = T * pnt;
    Eigen::Vector3f warped_points = cam2pixel(proj_pnt/proj_pnt[2], K);

    Eigen::MatrixXf tmp_J(1, 2);
    float d = proj_pnt[2];
    float warped_d = 0.0;
    interpolate_with_jacobian(depth_img, warped_points[0], warped_points[1], &warped_d, tmp_J);

    float ori_err = std::pow((d - warped_d/5000.0), 2);

    Eigen::Vector3f tmp_pnt = pnt;
    for(int i = 0; i < 3; i++){
        tmp_pnt[i] += h;
        proj_pnt = T * tmp_pnt;
        warped_points = cam2pixel(proj_pnt/proj_pnt[2], K);
        
        
        interpolate_with_jacobian(depth_img, warped_points[0], warped_points[1], &warped_d, tmp_J);
        float res_err = std::pow(d-warped_d/5000.0, 2);
        J_pnt(0, i) =  (res_err - ori_err)/h;
        tmp_pnt[i] = pnt[i];
    } 
   

    Eigen::VectorXf se3 = T.log();
    Eigen::VectorXf se3_tmp = se3;

    for(int i = 0; i < 6; i++){
        se3_tmp[i] += h;

        proj_pnt = Sophus::SE3f::exp(se3_tmp) * pnt;
        warped_points = cam2pixel(proj_pnt/proj_pnt[2], K);
        
        warped_d = 0.0;
        interpolate_with_jacobian(depth_img, warped_points[0], warped_points[1], &warped_d, tmp_J);
        float res_err = std::pow(d - warped_d/5000.0, 2);
        J_T(0, i) =  (res_err - ori_err)/h;
        se3_tmp[i] = se3[i];
    }

}