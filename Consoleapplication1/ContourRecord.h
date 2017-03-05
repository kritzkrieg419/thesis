#pragma once

#include <opencv2/core/core.hpp>
#include <boost/thread.hpp>

namespace ContourRecord {
	void save_hand(unsigned short* depthBuffer, cv::Mat depthMat, cv::Mat colorMat);
	void show_recording(cv::Mat* pColorMat);
	void reset_recording();
	void save_frame();
	void save_recording(int);
	void load_recording(int);

}