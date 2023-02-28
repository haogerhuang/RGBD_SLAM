#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <cstdlib>
#include <random>


#include "include/rgbd.h"
#include "include/slam.h"


int main(int argc, char *argv[]){
    

    std::string dataset_name = "room1";
    Eigen::Matrix3f K;
    // K << 380.678,0,318.402,0,380.678,239.805,0,0,1;

    K << 600.0, 0, 599.5, 0, 600.0, 339.5, 0, 0, 1;
    float depth_factor = 5000.0;

    //std::vector<float> scale_list = {0.5, 0.25, 0.125, 0.0625};
    std::vector<float> scale_list = {0.8, 0.15, 0.075};

    //Sophus::SE3f running_pose;

    int start;
    int end;

    int ttl_frames;

    std::cout<<"Start:\n";
    std::cin>>start;
    std::cout<<"End\n";
    std::cin>>end;

    Slam my_slam;


    pcl::visualization::PCLVisualizer viewer ("Real-time Point Cloud Viewer");

    // Initialize the viewer
    viewer.initCameraParameters ();
    viewer.setBackgroundColor (0, 0, 0);

    // Create a point cloud object

    // Add the point cloud to the viewer

    int i = start;
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(my_slam.cloud);
    pcl::visualization::PointCloudGeometryHandlerXYZ<pcl::PointXYZRGB> geo(my_slam.cloud);
    viewer.addPointCloud<pcl::PointXYZRGB> (my_slam.cloud, rgb, "point cloud");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "point cloud");

    // Start the viewer
    while (!viewer.wasStopped ())
    {
        // Generate the point cloud data in real-time using RGBD SLAM system
        // ...
        if ( i < end){
            RGBD data(dataset_name, i, scale_list, depth_factor);
            data.setK(K);
            my_slam.insert(data);
            i++; 
        }

      // Update the point cloud in the viewer
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_ud(my_slam.cloud);
      viewer.updatePointCloud<pcl::PointXYZRGB> (my_slam.cloud, rgb_ud,  "point cloud");

      // Update the viewer
      viewer.spinOnce ();
    }

	return 0;
}


