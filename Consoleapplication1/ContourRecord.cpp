#include <vector>
#include "ContourRecord.h"
#include "Isolines.h"
#include <iostream>
#include <numeric>
#include <fstream>
#include <boost/thread.hpp>
#include "Grabber.h"
#include <string>
#include <boost/tokenizer.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace ContourRecord {
	using namespace std;
	std::vector<std::vector<cv::Point>> recorded_frames;
	std::vector<std::vector<cv::Vec4b>> recorded_frames_vecs;
	int current_frame = 0;
	std::vector<cv::Point> previous_points;
	std::vector<cv::Vec3b> previous_vecs;


	void reset_recording() {
	}

	void save_frame() {

	}

	void load_recording(int name) {
		cout << "LOAD RECORDING" << endl;
		recorded_frames.clear();
		recorded_frames_vecs.clear();

		std::vector<std::vector<int>> xs;
		std::vector<std::vector<int>> ys;
		std::vector<std::vector<std::vector<int>>> vecs_int;
		
		std::string text1 = "xs.ser";
		std::string text2 = std::to_string(name);
		std::string text3 = text1 + text2;
		std::ifstream ifs(text3);
		boost::archive::text_iarchive iax(ifs);
		iax & xs;

		text1 = "ys.ser";
		text2 = std::to_string(name);
		text3 = text1 + text2;
		std::ifstream ifsy(text3);
		boost::archive::text_iarchive iay(ifsy);
		iay & ys;

		text1 = "vecs.ser";
		text2 = std::to_string(name);
		text3 = text1 + text2;
		std::ifstream ifsv(text3);
		boost::archive::text_iarchive iav(ifsv);
		iav & vecs_int;

		for (size_t frame_it = 0; frame_it < xs.size(); frame_it++) {
			std::vector<int> x = xs[frame_it];
			std::vector<int> y = ys[frame_it];
			std::vector<std::vector<int>> vecs_int_frame = vecs_int[frame_it];
			std::vector<cv::Point> frame;
			std::vector<cv::Vec4b> frame_vecs;


			for (size_t i = 0; i < x.size(); i++) {
				cv::Point point(x[i], y[i]);
				std::vector<int> vecs_int_pixel = vecs_int_frame[i];
				cv::Vec4b vec = { (unsigned char)vecs_int_pixel[0],(unsigned char)vecs_int_pixel[1],(unsigned char)vecs_int_pixel[2],(unsigned char)vecs_int_pixel[3] };
				frame.push_back(point);
				frame_vecs.push_back(vec);
			}
			recorded_frames.push_back(frame);
			recorded_frames_vecs.push_back(frame_vecs);
		}
	}

	void save_recording(int name) {
		
		std::vector<std::vector<int>> xs;
		std::vector<std::vector<int>> ys;
		std::vector<std::vector<std::vector<int>>> vec_vec_vec_int;

		std::vector<std::vector<cv::Vec4b>>::iterator vec_it, vec_end;
		std::vector<std::vector<cv::Point>>::iterator it, end;
		for (it = recorded_frames.begin(), vec_it = recorded_frames_vecs.begin(), vec_end = recorded_frames_vecs.end(); it != recorded_frames.end(); it++, vec_it++) {
		

			//Convert 2d vector of cv::vecs to 3d vector of ints
			std::vector<cv::Vec4b> vecs = *vec_it;  //Vector of cv::Vecs for one frame
			std::vector<std::vector<int>> vec_vec_int; //Vector of vector of ints for one frame

			//Loop through vecs and fill out vec_vec_int
			for (std::vector<cv::Vec4b>::iterator inner_vec_it = vecs.begin(); inner_vec_it != vecs.end(); inner_vec_it++) {
				cv::Vec4b cv_vec = *inner_vec_it;
				std::vector<int> temp_vec_int = { cv_vec[0], cv_vec[1], cv_vec[2], cv_vec[3] };
				vec_vec_int.push_back(temp_vec_int);
			}

			//We have one frame of ints, push frame back to vector of frames
			vec_vec_vec_int.push_back(vec_vec_int);
			

			//Convert cv::points to ints
			std::vector<cv::Point> points = *it;
			std::vector<int> x;
			std::vector<int> y;
			for (std::vector<cv::Point>::iterator inner_it = points.begin(); inner_it != points.end(); inner_it++) {
				cv::Point point = *inner_it;
				x.push_back(point.x);
				y.push_back(point.y);
			}
			xs.push_back(x);
			ys.push_back(y);
			
		}

		std::string text1 = "xs.ser";
		std::string text2 = std::to_string(name);
		std::string text3 = text1 + text2;
		std::ofstream ofsx(text3);
		boost::archive::text_oarchive oax(ofsx);
		oax & xs;

		text1 = "ys.ser";
		text2 = std::to_string(name);
		text3 = text1 + text2;
		std::ofstream ofsy(text3);
		boost::archive::text_oarchive oay(ofsy);
		oay & ys;

		text1 = "vecs.ser";
		text2 = std::to_string(name);
		text3 = text1 + text2;
		std::ofstream ofsv(text3);
		boost::archive::text_oarchive oav(ofsv);
		oav & vec_vec_vec_int;

	}

	void save_hand(unsigned short* depthBuffer, cv::Mat depthMat, cv::Mat colorMat) {
		std::vector<cv::Point> recorded_points;
		std::vector<cv::Vec4b> recorded_points_vecs;

		ICoordinateMapper* coordinateMapper = Grabber::get_pCoordinateMapper();
		cv::MatIterator_<cv::Vec3b> it, end;
		int depthBufferIt = 0;
		for (it = depthMat.begin<cv::Vec3b>(), end = depthMat.end<cv::Vec3b>(); it != end; it++) {
			cv::Vec3b current_pixel = *it;
			//If curret pixel is blue
			if (current_pixel[0] > 254 && current_pixel[1] == 0 && current_pixel[2] == 0) {
				//This depth pixel is a hand
				cv::Point current_depth_pos = it.pos();

				//Map pixel to color space
				DepthSpacePoint depthSpacePoint = { static_cast<float>(current_depth_pos.x), static_cast<float>(current_depth_pos.y) };
				ColorSpacePoint colorPoint;
				UINT16 depth = depthBuffer[current_depth_pos.y * 512 + current_depth_pos.x];
				HRESULT hr = coordinateMapper->MapDepthPointToColorSpace(depthSpacePoint, depth, &colorPoint);
				if (SUCCEEDED(hr)) {
					if (colorPoint.X < 1920 && colorPoint.X > 0 && colorPoint.Y < 1080 && colorPoint.Y > 0) {
						//Depth pixel containing hand has been successfully mapped to color pixel
						recorded_points.push_back(cv::Point(colorPoint.X, colorPoint.Y));
						recorded_points_vecs.push_back(colorMat.at<cv::Vec4b>(cv::Point(colorPoint.X, colorPoint.Y)));
						
						if (colorPoint.Y + 1 < 1080 && colorPoint.Y - 1 > 0) {
							recorded_points.push_back(cv::Point(colorPoint.X, colorPoint.Y + 1));
							recorded_points.push_back(cv::Point(colorPoint.X, colorPoint.Y - 1));
							recorded_points_vecs.push_back(colorMat.at<cv::Vec4b>(cv::Point(colorPoint.X, colorPoint.Y - 1)));
							recorded_points_vecs.push_back(colorMat.at<cv::Vec4b>(cv::Point(colorPoint.X, colorPoint.Y + 1)));
						}
					}
				}
			}

			//If current pixel is yellow
			if (current_pixel[0] == 0 && current_pixel[1] > 254 && current_pixel[2] > 254) {
				//This depth pixel is hand or the object the hand is grasping

			}

			depthBufferIt++;
		}

		recorded_frames.push_back(recorded_points);
		recorded_frames_vecs.push_back(recorded_points_vecs);
	}

	void show_recording(cv::Mat* pColorMat) {
		
		//Return immediately if no recorded frames are present
		if (recorded_frames.size() == 0) {
			return;
		}


		

		//Increment frame counter
		current_frame++;

		//If we have exhausted recoreded frames, start over
		if (current_frame >= static_cast<int>(recorded_frames.size())) {
			current_frame = -60;
		}

		if (current_frame < 0) {
			current_frame++;
			return;
		}
		//cout << current_frame << endl;

		//Get vector of handpoints for current frame
		std::vector<cv::Point> recorded_points1 = recorded_frames[current_frame];
		std::vector<cv::Vec4b> recorded_vecs1 = recorded_frames_vecs[current_frame];

		
		//Return if vector contains no points
		if (recorded_points1.size() < 1) {
			return;
		}


		//Loop through vector, map to color space and draw pixels
		std::vector<cv::Point>::iterator it, end;
		std::vector<cv::Vec4b>::iterator vec_it, vec_end;
		cv::Mat zeros = cv::Mat::zeros(pColorMat->size(), CV_8UC4);
		
		for (it = recorded_points1.begin(), end = recorded_points1.end(), vec_it = recorded_vecs1.begin(), vec_end = recorded_vecs1.end(); it != end; it++, vec_it++) {
			cv::Point point = *it;
			cv::Vec4b vec = *vec_it;
			if (point.x > 0 && point.x < 1920 && point.y > 0 && point.y < 1080) {
				cv::Vec4b thePixel = pColorMat->at<cv::Vec4b>(*it);
				cv::Vec4b greenPixel = thePixel*0.3 + cv::Vec4b(0, 255, 0, 255)*0.7;
				cv::Vec4b newPixel = thePixel*0.3 + vec*0.7;
				
				pColorMat->at<cv::Vec4b>(*it) = newPixel;
				zeros.at<cv::Vec4b>(*it) = vec;
			}
		}

		//cv::namedWindow("Hand Pixels");
		//cv::imshow("Hand Pixels", zeros);
	}

	std::vector<cv::Point> get_points() {
	}

	void save_hand2(unsigned short* depthBuffer, cv::Mat depthMat, cv::Mat colorMat) {
		std::vector<cv::Point> recorded_points;
		std::vector<cv::Vec4b> recorded_points_vecs;
		DepthSpacePoint* m_pDepthCoordinates = new DepthSpacePoint[1920 * 1080];

		ICoordinateMapper* coordinateMapper = Grabber::get_pCoordinateMapper();

		HRESULT hr = coordinateMapper->MapColorFrameToDepthSpace(512 * 424, (UINT16*)depthBuffer, 1920 * 1080, m_pDepthCoordinates);

		if (SUCCEEDED(hr)) {
			for (int colorIndex = 0; colorIndex < (1920 * 1080); ++colorIndex) {
				

				int colorX = colorIndex % 1920;
				int colorY = (colorIndex - colorX) / 1920;
				DepthSpacePoint p = m_pDepthCoordinates[colorIndex];

				//Check the depth-point for sanity
				if (p.X != std::numeric_limits<float>::infinity() && p.Y != std::numeric_limits<float>::infinity()) {
					int depthX = static_cast<int>(p.X + 0.5f);
					int depthY = static_cast<int>(p.Y + 0.5f);

					if (depthX > 0 && depthX < 1920 && depthY > 0 && depthY < 1080) {
						cv::Vec3b depth_pixel = depthMat.at<cv::Vec3b>(cv::Point(depthX, depthY));
						if (depth_pixel[0] > 254) {
							//This depth pixel is part of the hand
							recorded_points.push_back(cv::Point(colorX, colorY));
							recorded_points_vecs.push_back(colorMat.at<cv::Vec4b>(cv::Point(colorX, colorY)));
						}
					}
				}
			}
			recorded_frames.push_back(recorded_points);
			recorded_frames_vecs.push_back(recorded_points_vecs);
		}

		
	}


}