#pragma once
#include <opencv2/core/core.hpp>
#include <kinect.h>


namespace Grabber {
	
	void grabber_init();
	void grabber_start();
	void frame_acquisition();
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertRGBDepthToPointXYZRGB();

	unsigned short* get_p_depth_buffer();
	ICoordinateMapper* get_pCoordinateMapper();
	cv::Mat get_depth_norm_mat();
	cv::Mat get_depth_mat();
	cv::Mat get_color_mat();
	ColorSpacePoint getLeftHandWrist();
	ColorSpacePoint getLeftHandTip();
	ColorSpacePoint getLeftHandCentroid();
	ColorSpacePoint getLeftHandThumb();
	unsigned char* getBodyIndexBuffer();
	int getBodyIndexBufferLength();
	unsigned short* getDepthBuffer();
	std::vector<DepthSpacePoint> getColorToDepthMapping();
	bool isTrackingBody();
	void processBodies(const unsigned int &bodyCount, IBody **bodies);
	unsigned short* smoothDepthArray(unsigned short* depthArray, int innerBand, int outerBand);
	cv::Mat get_depth_norm_contour_mat();
}
