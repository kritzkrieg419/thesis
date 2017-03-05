// Disable Error C4996 that occur when using Boost.Signals2.
#ifdef _DEBUG
#define _SCL_SECURE_NO_WARNINGS
#endif

#define MAX_DEPTH 1000

#include "BackgroundRemoval.h"
#include "Preprocessing.h"
#include <boost/thread.hpp>
#include "Grabber.h"
#include "HandTracking.h"
#include "ContourRecord.h"
#include <unordered_map>
#include <boost/format.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <deque>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
int num_callbacks;
Eigen::Vector4f first_loc;
cv::Mat old_target_mat;
int old_target_mat_x;
int old_target_mat_y;
int frame_counter;
std::vector<cv::Point2f> upperLeftCorners;
cv::Mat prev_frame;
cv::Mat canny_output;
std::vector<std::vector<cv::Point>> contours;
std::vector<cv::Vec4i> hierarchy;
cv::Mat depth_img_gray_eight;
cv::Mat drawing;
std::vector<cv::Point> approxShape;
cv::Mat depth_img_gray;
cv::Mat img_color;
cv::Mat frame_difference;
DepthSpacePoint handSeedPoint;
UINT16 depth;
unsigned short* depth_buffer;
bool grab = false;
int count = 0;
bool madeSnapshot = false;
cv::Mat snapshot;
std::vector<int> handXs;
std::vector<int> handYs;
std::vector<cv::Point> snapshot_points;
std::vector<cv::Point> colorPoints;
std::vector<cv::Vec4b> snapshot_pixels;
int backgroundSet = 0;
bool doTheSnap = false;
cv::Mat backgroundMat;
cv::Rect button;
int current_cycle = 0;
bool playback_mode = false;
std::string text;
std::string text2;
std::string text1;
std::string text21;
std::string text22;
std::string text3;
std::deque<unsigned short*> averageQueue;
unsigned short* smoothed_depth = new unsigned short[512 * 424];
std::vector<int> frametimes;

template <class Interface> inline void safe_release(Interface **ppT)
{
	if (*ppT)
	{
		(*ppT)->Release();
		*ppT = NULL;
	}
}



void callBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == 1)
	{
		if (y > 980) {
			//Clicking buttons
			if (x < 100)
				//first button
				current_cycle = 10;
			else if (x < 200)
				//second button
				current_cycle = 11;
			else if (x < 300)
				//etc
				current_cycle = 12;
			else if (x < 400)
				current_cycle = 13;
			else if (x < 500)
				current_cycle = 14;
			else if (x < 600)
				current_cycle = 15;
			else if (x < 700)
				current_cycle = 16;
			return;

		}
		if (true)
		{
			cout << "Playback loaded" << endl;
			ContourRecord::load_recording(current_cycle);
		}
	}

	if (event == 2) {
		if (true) {
			cout << "Playback saved" << endl;
			ContourRecord::save_recording(current_cycle);
		}
	}

	if (event == 3) {
		cout << "Cycling stored playback " << current_cycle << endl;
		current_cycle++;
		if (current_cycle > 7) {
			current_cycle = 0;
		}
	}

	if (event == 9) {
		if (playback_mode) {
			cout << "Playback mode OFF" << endl;
			playback_mode = !playback_mode;
		}
		if (!playback_mode) {
			cout << "Playback mode ON" << endl;
			playback_mode = !playback_mode;
		}
	}

}


int main()
{	
	_CrtSetDbgFlag(_CRTDBG_CHECK_ALWAYS_DF);

	//Initialize and start Grabber
	Grabber::grabber_init();
	Grabber::grabber_start();

	//Acquire two frames initially to make sure the sensor is 'hot'
	Grabber::frame_acquisition();
	Grabber::frame_acquisition();

	//OpenCV windows for visualization
	//cv::namedWindow("Difference Frame");
	//cv::namedWindow("Hand Contour");
	cv::namedWindow("Color frame");
	//cv::namedWindow("Depth buffer");
	int warmupcycles = 0;
	int sum = 0;
	while (1)
	{
		warmupcycles++;
		int new_hand_pixels = 0;
		int abandoned_hand_pixels = 0;
		int current_hand_pixels = 0;

		

		//Save current time at start of frame
		boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
		
		//Get next frame from Grabber
		Grabber::frame_acquisition();
		
		
		img_color = Grabber::get_color_mat();
		
		//Filter far objects
		Preprocessing::depth_passthrough(MAX_DEPTH);
		

		depth_buffer = Grabber::get_p_depth_buffer();
		depth_buffer = Preprocessing::temporal_smoothing(depth_buffer, averageQueue, 15);
		
		/*std::copy(smoothed_depth_p, smoothed_depth_p + (424 * 512), smoothed_depth);

		cv::Mat temp = cv::Mat(cv::Size(512, 424), CV_16UC1, smoothed_depth);
		cv::Mat depth_img_rgb_space;
		cv::Mat depth_img_rgb_space_norm;

		// convert grayscale to color image
		cv::cvtColor(temp, depth_img_rgb_space, cv::COLOR_GRAY2BGR, 3);

		//Normalize values in depth buffer array for better visualization
		double min;
		double max;
		cv::minMaxIdx(depth_img_rgb_space, &min, &max);
		cv::convertScaleAbs(depth_img_rgb_space, depth_img_rgb_space_norm, 65535 / max);
		*/
		
		//Spatial 'hole filling' noise reduction
		//depth_buffer = Preprocessing::smooth_depth(depth_buffer, 2, 5);
		
		//Backgorund filtering
		BackgroundRemoval::setBackground(Grabber::get_depth_norm_mat());
		
		
		//Convert depth buffer to 8bit 1 channel
		//cv::Mat masked_depth_norm = BackgroundRemoval::subtract(Grabber::get_depth_norm_mat());
		cv::Mat masked_depth_norm = Grabber::get_depth_norm_mat();
		//Hand recognition
		int foundHand = HandTracking::findHand(&handSeedPoint, &depth, masked_depth_norm);
		//int foundHand = 1;
		//Blur the masked depth norm
		//cv::GaussianBlur(masked_depth_norm, masked_depth_norm, cv::Size(0, 0), 1.5);

		cv::Mat depth_mat = Grabber::get_depth_mat();
		drawing = Preprocessing::find_contours(depth_mat, drawing);

		HandTracking::fill_hand(grab, foundHand, playback_mode, warmupcycles, handSeedPoint.X, handSeedPoint.Y, drawing);
		cv::Mat zeros = cv::Mat::zeros(drawing.size(), CV_8UC3);
		
		
		//If there is a hand in the image and we've processed 25 frames (making sure the sensor is warmed up)
		if (!foundHand && warmupcycles > 25 && !playback_mode) {

			//Draw circle to show handSeedPoint
			cv::circle(drawing, cv::Point(handSeedPoint.X, handSeedPoint.Y), 10, cv::Scalar(255, 255, 255), CV_FILLED, 8, 0);
			
			//Prepare difference frame
			frame_difference = HandTracking::frame_difference(prev_frame, drawing, &new_hand_pixels, &abandoned_hand_pixels);

			//Save current frame to prev_frame
			drawing.copyTo(prev_frame);
			
			//If new frame has 3500 new hand-pixels, hand is grabbing
			if (new_hand_pixels > 3500) {
				grab = true;
			}

			//If current frame has 3500 abandoned handpixels, hand is not grabbing
			if (abandoned_hand_pixels > 3500) {
					grab = false;
			}

			//cv::imshow("Difference Frame", frame_difference);
			//cv::waitKey(33);

	
			//Perform background subtraction on hand-pixels
			for (int x = 0; x != 512; x++) {
				for (int y = 0; y != 424; y++) {
					//Get current pixel in the masked depth buffer
					cv::Vec3b masked_pixel = masked_depth_norm.at<cv::Vec3b>(cv::Point(x, y));
					//If depth buffer pixel is part of non-background
					if (masked_pixel[0] == 255 && masked_pixel[1] == 255 && masked_pixel[2] == 255) {

						//Get current hand contour pixel
						cv::Vec3b drawing_pixel = drawing.at<cv::Vec3b>(cv::Point(x, y));

						//If current hand contour pixel is part of the hand, we have an actual hand pixel
						if (drawing_pixel[0] == 255 && drawing_pixel[1] == 0 && drawing_pixel[2] == 0) {
							zeros.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(255, 0, 0);
						}
					}
				}
			}

			//Save hand-frame in ContourRecord
			ContourRecord::save_hand(depth_buffer, zeros, img_color);
		}

		//Visualization
		cv::imshow("Hand Contour", drawing);		

		//If no hand is detected in image, overlay recording of previous hand movements
		//(If any exists in ContourRecord)
		if (foundHand || playback_mode) {
			ContourRecord::show_recording(&img_color);
		}

		//Measure time at end of frame
		boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
		
		//Calculate framtime
		boost::posix_time::time_duration diff = t2 - t1;
		int frametime_ms = static_cast<int>(diff.total_milliseconds());
		frametimes.push_back(frametime_ms);

		//Calculate average frametime
		
		if (frametimes.size() % 10 == 0) {
			if (frametimes.size() > 150) {
				frametimes.erase(frametimes.begin());
			}

			sum = 0;
			for (std::vector<int>::iterator i = frametimes.begin(); i != frametimes.end(); ++i) {
				sum += *i;
			}
			sum = sum / frametimes.size();
		}
		

	

		//Display depth frame
		//cv::Mat depthMat = Grabber::get_depth_norm_mat();
		//cv::imshow("Depth buffer", depthMat);

		//Display masked depth frame
		//cv::namedWindow("Masked depth buffer");
		//cv::imshow("Masked depth buffer", masked_depth_norm);

		//Display registered background
		//cv::namedWindow("Registered background");
		//cv::imshow("Registered background", BackgroundRemoval::getBackground());

		//Create buttons for choosing playback
		button = cv::Rect(0, 0, img_color.cols, 50);
		text = "Load playback M1, Save playback M2, Cycle playback M3, Playback mode double-M3, current playback: ";
		text2 = std::to_string(current_cycle);
		text21 = " Playback mode: ";
		text22 = std::to_string(playback_mode);
		std::string text3 = text + text2 + text21 + text22;

		//Set mouselistener on color frame for user interaction
		cv::setMouseCallback("Color frame", callBackFunc);

		//The color image mat is flipped, reverse it
		cv::Mat img_color_flipped;
		cv::flip(img_color, img_color_flipped, 1);
		
		//Draw buttons for 'wizard of oz' puzzle experiment
		//cv::rectangle(img_color_flipped, cv::Point(0, 1080 - 100), cv::Point(100, 1080), cv::Scalar(255, 255, 255));
		//cv::rectangle(img_color_flipped, cv::Point(100, 1080 - 100), cv::Point(200, 1080), cv::Scalar(255, 255, 255));
		//cv::rectangle(img_color_flipped, cv::Point(200, 1080 - 100), cv::Point(300, 1080), cv::Scalar(255, 255, 255));
		//cv::rectangle(img_color_flipped, cv::Point(300, 1080 - 100), cv::Point(400, 1080), cv::Scalar(255, 255, 255));
		//cv::rectangle(img_color_flipped, cv::Point(400, 1080 - 100), cv::Point(500, 1080), cv::Scalar(255, 255, 255));
		//cv::rectangle(img_color_flipped, cv::Point(500, 1080 - 100), cv::Point(600, 1080), cv::Scalar(255, 255, 255));
		//cv::rectangle(img_color_flipped, cv::Point(600, 1080 - 100), cv::Point(700, 1080), cv::Scalar(255, 255, 255));
		cv::putText(img_color_flipped, text3, cv::Point(button.width*0.35, button.height*0.7), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0));

		//Display the frametime in colorframe
		cv::putText(img_color_flipped, std::to_string(sum), cv::Point(50, 50), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0));

		//Display flipped color frame
		cv::imshow("Color frame", img_color_flipped);

		//Display masked hand contour (hand contour minus registered background)
		//cv::namedWindow("masked hand contour");
		//cv::imshow("masked hand contour", zeros);

		//cv::namedWindow("Smoothed depth");
		//cv::imshow("Smoothed depth", depth_img_rgb_space_norm);




		

		//Hold all displayed images for 33ms
		cv::waitKey(33);

	}
	return 0;
}


