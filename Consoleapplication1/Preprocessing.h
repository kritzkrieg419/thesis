#pragma once
#include <deque>

namespace Preprocessing {
	void depth_passthrough(int max_depth);

	unsigned short* smooth_depth(unsigned short* depth_buffer, int inner_band, int outer_band);

	cv::Mat find_contours(cv::Mat, cv::Mat);

	unsigned short* temporal_smoothing(unsigned short* current_frame, std::deque<unsigned short*> average_queue, int n_frames);


}