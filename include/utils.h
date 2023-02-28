#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Core>
#include <Eigen/LU>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>> VecVector2f;
typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> VecVector3f;
typedef std::vector<Vector6f, Eigen::aligned_allocator<Vector6f>> VecVector6f;

Eigen::Vector3f pixel2cam(const Eigen::Vector3f &p, const Eigen::Matrix3f &K);
Eigen::Vector3f cam2pixel(const Eigen::Vector3f &p, const Eigen::Matrix3f &K);
Eigen::Vector3f cam3d(const Eigen::Vector3f &p, const cv::Mat& depth_img);
Eigen::MatrixXf matmul_mat_diag(const Eigen::MatrixXf &mat1, const Eigen::VectorXf& mat2);
Eigen::VectorXf matmul_diag_vec(const Eigen::VectorXf& mat1, const Eigen::VectorXf& mat2);
bool check_identical(const Eigen::MatrixXf& mat1, const Eigen::MatrixXf& mat2);
Eigen::VectorXf diag_inverse(const Eigen::VectorXf &mat);

Sophus::SE3f kabsh_algorithm(
    const VecVector3f& pnts1_3d,
    const VecVector3f& pnts2_3d
);

void numerical_jacobian(const Sophus::SE3f& T1, const Sophus::SE3f& T2, const Eigen::Vector3f& pnt, Eigen::MatrixXf& Jacobian, float h=1e-6);
void numerical_jacobian(const Sophus::SE3f& T, const Eigen::Vector3f& pnt, const Eigen::Matrix3f& K,
                        const Eigen::Vector3f& rgb, const cv::Mat& rgb_img, 
                        Eigen::MatrixXf& J_T, Eigen::MatrixXf& J_pnt, float h = 1e-6);

void numerical_jacobian(const Sophus::SE3f& T, const Eigen::Vector3f& pnt, const Eigen::Matrix3f& K,
                        const cv::Mat& depth_img, Eigen::MatrixXf& J_T, Eigen::MatrixXf& J_pnt, float h = 1e-6);


float interpolate(const cv::Mat& img, float x, float y, bool print=false);
void interpolate_with_jacobian(const cv::Mat& img, float x, float y, Eigen::Vector3f& res, Eigen::MatrixXf& Jacobian);
void interpolate_with_jacobian(const cv::Mat& img, float x, float y, float* res, Eigen::MatrixXf& Jacobian);

#endif
