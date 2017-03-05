#include "HandTracking.h"

#include <boost/format.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include "Grabber.h"
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>

#include <unordered_map>
#include <algorithm>
#define MAX_DEPTH 1000
#define PIXEL_THRESHOLD 30


namespace HandTracking
{
	std::vector<DepthSpacePoint> hand_depth_points;
	std::vector<int> contourVec;
	std::set<int> xs;
	std::set<int> ys;
	unsigned short* global_depth_buffer;
	bool* contourArr = new bool[424 * 512];
	bool* prev_handContourArr = new bool[424 * 512];
	bool* handContourArr = new bool[424 * 512];
	int testcount = 0;


	/**
	* Given (x,y) return index for (x,y) point in buffer
	*/
	int lookupHelper(int x, int y)
	{
		int depthWidth = 512;
		return y*depthWidth + x;
	}

	int lookupModifier(int index, int x_mod, int y_mod)
	{
		int depthWidth = 512;
		return index + (y_mod*depthWidth + x_mod);
	}
	

	/** For each pixel in curr_frame, if candidate pixel is blue (is_hand) check if same pixel in prev_frame is blue
		if not, then candidate pixel is a new hand-pixel and should be added to difference frame
		return difference frame
	*/
	cv::Mat frame_difference(cv::Mat prev_frame, cv::Mat curr_frame, int* new_hand_pixels, int* former_hand_pixels) {
		if (prev_frame.size() != curr_frame.size()) {
			return cv::Mat::zeros(curr_frame.size(), CV_8UC3);
		}	

		cv::Mat difference_frame = cv::Mat::zeros(curr_frame.size(), CV_8UC3);
		
		//Loop through pixels in Mat
		int new_pixels = 0;
		int old_pixels = 0;
		for (int x = 0; x < curr_frame.size().width; x++) {
			for (int y = 0; y < curr_frame.size().height; y++) {
				cv::Vec3b candidate_pixel = curr_frame.at<cv::Vec3b>(cv::Point(x, y));
				cv::Vec3b prev_pixel = prev_frame.at<cv::Vec3b>(cv::Point(x, y));

				//Set up predicates that check if candidate pixel is blue or yellow
				bool predicate_candidate = (candidate_pixel[0] > 254 && candidate_pixel[1] == 0 && candidate_pixel[2] == 0 ||
					(candidate_pixel[0] == 0 && candidate_pixel[1] > 254 && candidate_pixel[2] > 254));

				bool predicate_prev = (prev_pixel[0] > 254 && prev_pixel[1] == 0 && prev_pixel[2] == 0 ||
					(prev_pixel[0] == 0 && prev_pixel[1] > 254 && prev_pixel[2] > 254));

				//Check if candidate pixel is blue or yellow
				if (predicate_candidate) {
				
					//Check that candidate pixel's previous self was NOT blue or yellow
					if (!predicate_prev) {
						//Current pixel IS HAND, previous pixel WAS NOT hand. Color blue (NEW HAND)
						difference_frame.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(255, 0, 0);
						new_pixels++;
					}
				}

				//Check if candidate pixel is NOT blue or yellow
				if (!predicate_candidate) {
					
					//Check if candidate pixel's previous self WAS blue or yellow
					if (predicate_prev) {
						//Current pixel is NOT hand, previous pixel WAS HAND. Color yellow (OLD HAND)
						difference_frame.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(0, 255, 255);
						old_pixels++;
					}
				}
			}
		}
		*new_hand_pixels = new_pixels;
		*former_hand_pixels = old_pixels;
		return difference_frame;
	}



	bool find_depthspacepoint_in_vector(std::vector<DepthSpacePoint> vec, DepthSpacePoint point) {
		for (std::vector<DepthSpacePoint>::iterator it = vec.begin(); it != vec.end(); it++) {
			DepthSpacePoint current_point = *it;
			if (current_point.X == point.X && current_point.Y == point.Y) {
				return true;
			}
		}
		return false;
	}

	void fill_hand(bool grab, bool foundHand, bool playback_mode, int warmupcycles, int handSeedPoint_X, int handSeedPoint_Y, cv::Mat drawing) {
		cv::Scalar floodfill_color = cv::Scalar();
		//Prepare floodfill color
		if (!grab) {
			floodfill_color = cv::Scalar(255, 0, 0); //Not grabbing anything: Blue
		}
		if (grab) {
			//floodfill_color = cv::Scalar(0, 255, 255); //Grabbing something: Yellow
			floodfill_color = cv::Scalar(255, 0, 0);
		}


		//If hand is detected in image, floodfill hand-contour and update difference frame
		if (!foundHand && warmupcycles > 25 && !playback_mode) {
			if (handSeedPoint_X > 500) {
				//Hand has entered from the sensor's right hand side
				cv::floodFill(drawing, cv::Point(handSeedPoint_X - 5, handSeedPoint_Y), floodfill_color);
			}
			if (handSeedPoint_X < 500) {
				//Hand has entered from the sensor's left hand side
				cv::floodFill(drawing, cv::Point(handSeedPoint_X + 5, handSeedPoint_Y), floodfill_color);
			}
		}
	}


	int findHand(DepthSpacePoint* depthSpacePoint, UINT16* depth, cv::Mat maskedDepthMat) {
		int depthWidth = 512;
		int depthHeight = 424;
		unsigned short* depthBuffer = Grabber::get_p_depth_buffer();

		//Loop through edge-pixels and check depth
		unsigned int intersecting_pixels_left = 0;
		unsigned int intersecting_pixels_right = 0;
		std::vector<unsigned int> ys_left;
		std::vector<unsigned int> ys_right;

		for (unsigned int y = 0; y < depthHeight; y++) {
			UINT16 depth_left = depthBuffer[y * depthWidth + 2];
			UINT16 depth_right = depthBuffer[y * depthWidth + 500];

		

			if (depth_left > 0 && depth_left < MAX_DEPTH) {
				//Something intersects the edge of the frame within 1m from sensor. 
				cv::Vec3b maskedLeftPixel = maskedDepthMat.at<cv::Vec3b>(cv::Point(2, y));
				if (maskedLeftPixel[0] == 0 && maskedLeftPixel[1] == 0 && maskedLeftPixel[2] == 0) {
					depth_left = 0;
				}
				else {
					ys_left.push_back(y);
					intersecting_pixels_left++;
				}
			}

			if (depth_right > 0 && depth_right < MAX_DEPTH) {
					cv::Vec3b maskedRightPixel = maskedDepthMat.at<cv::Vec3b>(cv::Point(500, y));
					if (maskedRightPixel[0] == 0) {
						depth_right = 0;
					}
					else {
						ys_right.push_back(y);
						intersecting_pixels_right++;
					}
			}
		}
		if (intersecting_pixels_left > PIXEL_THRESHOLD) {
			//Hand detected on the left
			unsigned int retval = ys_left[floor(intersecting_pixels_left / 2)];
			*depthSpacePoint = { static_cast<float>(2), static_cast<float>(retval) };
			*depth = depthBuffer[retval * depthWidth + 2];
			return 0;

		}

		if (intersecting_pixels_right > PIXEL_THRESHOLD) {
			//Hand detected on the right
			unsigned int retval = ys_right[floor(intersecting_pixels_right / 2)];
			*depthSpacePoint = { static_cast<float>(depthWidth - 3), static_cast<float>(retval) };
			*depth = depthBuffer[retval * depthWidth + depthWidth - 3];
			return 0;
		}

		*depthSpacePoint = { static_cast<float>(-1), static_cast<float>(-1) };
		*depth = 0;
		return -1;

	}


}