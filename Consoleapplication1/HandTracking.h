#pragma once
#include <Eigen/Dense>
#include <boost/thread.hpp>
#include <opencv2/core/core.hpp>
#include <kinect.h>
namespace HandTracking {
	int findHand(DepthSpacePoint* depthSpacePoint, UINT16* depth, cv::Mat maskedDepthMat);
	std::vector<int> contourHand(int threshold, int maxDepth);
	int lookupHelper(int x, int y);
	cv::Mat frame_difference(cv::Mat frame1, cv::Mat frame2, int* new_hand_pixels, int* former_hand_pixels);
	void fill_hand(bool grab, bool foundHand, bool playback_mode, int warmupcycles, int handSeedPoint_X, int handSeedPoint_Y, cv::Mat drawing);
}
