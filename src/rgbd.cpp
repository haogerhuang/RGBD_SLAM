#include "../include/rgbd.h"

std::vector<cv::KeyPoint> RGBD::get_keypoints(){
    if (keypoints.size() > 0) return keypoints;
 	cv::Ptr<cv::FeatureDetector> Detector = cv::SIFT::create();
	cv::Ptr<cv::DescriptorExtractor> Descriptor = cv::SIFT::create();
   
    Detector->detect(img_list[0], keypoints);
    Descriptor->compute(img_list[0], keypoints, descriptors);

    for(int i = 0; i < keypoints.size(); i++){
        kp2world.push_back(-1);
    }

    return keypoints;
}
Eigen::Vector3f RGBD::get_kp_by_id(int i){
    

    cv::Point2f p = keypoints[i].pt;
    Eigen::Vector3f pnt;
    pnt[0] = p.x;
    pnt[1] = p.y;
    pnt[2] = 1;
    return pnt;
}

VecVector6f RGBD::get_pc(int s){
    if (point_cloud.size() > 0) return point_cloud;

    int rows = img_list[s].rows;
    int cols = img_list[s].cols;

    for (int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            Vector6f p;

            p.head(3) = get_xyz(j, i, s);
            // Eigen::Vector3f rgb = get_rgb(j, i, s);
           
            p.tail(3) = get_rgb(j, i, s);

            point_cloud.push_back(p);
        }
    }
    return point_cloud;
    
}

VecVector3f RGBD::get_XYZ(int s){
    if (point_cloud_xyz.size() > 0) return point_cloud_xyz;

    int rows = img_list[s].rows;
    int cols = img_list[s].cols;

    for (int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            Eigen::Vector3f p = get_xyz(j, i, s);

            // p.tail(3) = get_rgb(j, i, s);

            point_cloud_xyz.push_back(p);
        }
    }
    return point_cloud_xyz;
    
}

void RGBD::setK(Eigen::Matrix3f _K){
    if (!K_list.empty()) return;
    for(double s: scales){
        Eigen::Matrix3f K = _K;
        K(0,0) = K(0,0)*s;
        K(1,1) = K(1,1)*s;
        K(0,2) = K(0,2)*s;
        K(1,2) = K(1,2)*s;
        K_list.push_back(K);
    }
}

Eigen::Matrix3f RGBD::getK(int s){
    return K_list[s];
}

cv::Mat RGBD::get_descriptors(){
    if (!descriptors.empty()) return descriptors;
    RGBD::get_keypoints();
    return descriptors;
}

Eigen::Vector3f RGBD::get_xyz(int x, int y, int s){
    if (x < 0 || x >= img_list[s].cols || y < 0 || y >= img_list[s].rows){
        Eigen::Vector3f xyz(0, 0, -1);
        return xyz;
    }
    Eigen::Vector3f xyz(x, y, 1);
    xyz = pixel2cam(xyz, K_list[s]); //camera
    float d = (float)depth_list[s].at<uint16_t>(y, x);

    d /= depth_factor;
    xyz = xyz*d;
    return xyz;
}

Eigen::Vector3f RGBD::get_rgb(int x, int y, int s){
    Eigen::Vector3f rgb;
    if (x < 0 || x >= img_list[s].cols || y < 0 || y >= img_list[s].rows){
        Eigen::Vector3f rgb(0, 0, -1);
        return rgb;
    }
    int step = img_list[s].step;
    int channels = img_list[s].channels();
    rgb[0] = (float)img_list[s].data[y * step + x * channels+0];
    rgb[1] = (float)img_list[s].data[y * step + x * channels+1];
    rgb[2] = (float)img_list[s].data[y * step + x * channels+2];
    return rgb;
}

//Eigen::Vector3f RGBD::get_rgb(float x, float y, int s){
//    Eigen::Vector3f rgb;
//    cv::Mat img = img_list[s];
//    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows){
//        Eigen::Vector3f rgb(0, 0, -1);
//        return rgb;
//    }
//    else{
//    }
//}

float RGBD::get_intensity(int x, int y, int s){
    Eigen::Vector3f rgb = get_rgb(x, y, s);
    float intensity = (rgb[0] + rgb[1] + rgb[2])/3;
    return intensity;
}

float RGBD::get_fx(int s){
    return K_list[s](0,0);
}
float RGBD::get_fy(int s){
    return K_list[s](1,1);
}

int RGBD::get_rows(int s){
    return img_list[s].rows;
}

int RGBD::get_cols(int s){
    return img_list[s].cols;
}

cv::Mat RGBD::get_img(int s){
    return img_list[s];
}

cv::Mat RGBD::get_gray(int s){
    cv::Mat gray;
    cv::cvtColor(img_list[s], gray, cv::COLOR_RGB2GRAY);
    return gray;
}

cv::Mat RGBD::get_depth(int s){
    return depth_list[s];
}

Sophus::SE3f RGBD::get_pose(){
    return pose;
}

void RGBD::set_pose(Sophus::SE3f pose_){
    pose = pose_;
}

std::pair<cv::Mat, cv::Mat> RGBD::get_img_gradient(){
    int s = scales.size()-1;
    if(need_GN_data) compute_GN_data(s);
    return img_gradient;
}

std::pair<cv::Mat, cv::Mat> RGBD::get_depth_gradient(){
    int s = scales.size()-1;
    if(need_GN_data) compute_GN_data(s);
    return depth_gradient;
}

cv::Mat RGBD::get_normal(){
    int s = scales.size()-1;
    if(need_GN_data) compute_GN_data(s);
    return normal;
}

void RGBD::compute_GN_data(int s){
    if (!need_GN_data) return;
    need_GN_data = false;

    cv::Mat img = get_gray(s);
    cv::Mat depth = get_depth(s);

    cv::Mat img_gradient_x;
    cv::Mat img_gradient_y;

    cv::Mat depth_gradient_x;
    cv::Mat depth_gradient_y;

    compute_Gradient(img, img_gradient_x, img_gradient_y);
    img_gradient = {img_gradient_x, img_gradient_y};

    compute_Gradient(depth, depth_gradient_x, depth_gradient_y);
    depth_gradient = {depth_gradient_x, depth_gradient_y};

    compute_Normal(depth_gradient_x, depth_gradient_y);

}

void RGBD::compute_Gradient(cv::Mat &img, cv::Mat &gradient_x, cv::Mat &gradient_y){

    int h = img.rows;
    int w = img.cols;

    gradient_x = cv::Mat::zeros(h, w, CV_32FC1);
    gradient_y = cv::Mat::zeros(h, w, CV_32FC1);

    for(int y = 0; y < h; y++){ 
        for(int x = 1; x < w-1; x++){
            float intensity1;
            float intensity2;
            if (img.type() == 0){
                intensity1 = (float)img.at<uint8_t>(y, x-1);
                intensity2 = (float)img.at<uint8_t>(y, x+1);
            }
            else if(img.type() == 2){
                intensity1 = (float)img.at<uint16_t>(y, x-1);
                intensity2 = (float)img.at<uint16_t>(y, x+1);
            }
            else{
                intensity1 = img.at<float>(y, x-1);
                intensity2 = img.at<float>(y, x+1);
            }

            gradient_x.at<float>(y, x) = 0.5f * (intensity2 - intensity1);
            //std::cout<<intensity2<<" "<<intensity1<<" "<<gradient_x.at<float>(y,x)<<"\n";
        }
    }

    for(int y = 1; y < h-1; y++){ 
        for(int x = 0; x < w; x++){
            float intensity1;
            float intensity2;
            if (img.type() == 0){
                intensity1 = (float)img.at<uint8_t>(y-1, x);
                intensity2 = (float)img.at<uint8_t>(y+1, x);
            }
            else if(img.type() == 2){
                intensity1 = (float)img.at<uint16_t>(y-1, x);
                intensity2 = (float)img.at<uint16_t>(y+1, x);
            }
            else{
                intensity1 = img.at<float>(y-1, x);
                intensity2 = img.at<float>(y+1, x);
            }
            gradient_y.at<float>(y, x) = 0.5f * (intensity2 - intensity1);
        }
    }
}

void RGBD::compute_Normal(cv::Mat& gradient_x, cv::Mat& gradient_y){
    int h = gradient_x.rows;
    int w = gradient_y.cols;

    normal = cv::Mat::zeros(h, w, CV_32FC3);
    for (int y = 0; y < h; y++){
        for(int x = 0; x < w; x++){
            normal.at<cv::Vec3f>(y, x)[0] = -gradient_x.at<float>(y, x);
            normal.at<cv::Vec3f>(y, x)[1] = -gradient_y.at<float>(y, x);
            normal.at<cv::Vec3f>(y, x)[2] = 1.0;
            float scale = cv::norm(normal.at<cv::Vec3f>(y, x), cv::NORM_L2);
            normal.at<cv::Vec3f>(y, x)[0] /= scale;
            normal.at<cv::Vec3f>(y, x)[1] /= scale;
            normal.at<cv::Vec3f>(y, x)[2] /= scale;
        }
    }
}
