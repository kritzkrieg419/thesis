#pragma once
#include <opencv2/highgui.hpp>
#include "opencv2/core/core.hpp"

namespace BackgroundRemoval {

	void update(cv::Mat);
	cv::Mat getMask();
	void setBackground(cv::Mat);
	cv::Mat subtract(cv::Mat);
	void learnBackground(cv::Mat);
	cv::Mat getBackground();
	bool readyToMask();
}
