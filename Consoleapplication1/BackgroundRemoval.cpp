// Disable Error C4996 that occur when using Boost.Signals2.
#ifdef _DEBUG
#define _SCL_SECURE_NO_WARNINGS
#endif
#include "BackgroundRemoval.h"
#include "stdafx.h"
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "Grabber.h"
#include <iostream>

namespace BackgroundRemoval {

	cv::Mat background;
	int backgroundSet = 0;
	bool pause = true;
	int input;

	cv::Mat subtract(cv::Mat matrix) {
		if (backgroundSet < 21) {
			return matrix;
		}
		cv::Mat temp;
		matrix.copyTo(temp);
		temp -= background;

		//Update depthbuffer
	/*	for (int x = 0; x < 512; x++) {
			for (int y = 0; y < 424; y++) {
				int i = y * 512 + x;
				unsigned short* depthBuffer = Grabber::getDepthBuffer();
				if (background.at<cv::Vec3b>()
				depthBuffer[i]
			}
		}*/

		return temp;
	}

	cv::Mat getBackground() {
		return background;
	}

	bool readyToMask() {
		if (backgroundSet > 21) {
			return true;
		}
		else {
			return false;
		}
	}

	void setBackground(cv::Mat frame) {
		if (backgroundSet > 21) {
			return;
		}

		backgroundSet++;
		if (backgroundSet < 20) {
			frame.copyTo(background);
		}

		//Dilate background image to compensate for random noise along edges
		if (backgroundSet == 21) {
			//cv::GaussianBlur(background, background, cv::Size(0, 0), 1.5);
			int erosion_size = 5;
			cv::Mat element = cv::getStructuringElement(cv::MORPH_DILATE,
				cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
				cv::Point(erosion_size, erosion_size));
			//cv::dilate(background, background, cv::Mat(), cv::Point(-1, -1));
			cv::dilate(background, background, element);
		}
	}




}