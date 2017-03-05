#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Preprocessing.h"
#include "Grabber.h"
#include <deque>
namespace Preprocessing {

	void depth_passthrough(int max_depth) {
		unsigned short* depth_buffer = Grabber::get_p_depth_buffer();
		for (int i = 0; i < 424 * 512; i++) {
			if (depth_buffer[i] > max_depth) {
				depth_buffer[i] = 0;
			}
		}
	}

	unsigned short* smooth_depth(unsigned short* depthArray, int innerBandThreshold, int outerBandThreshold) {
		unsigned short* smoothDepthArray = new unsigned short[512 * 424];
		int widthBound = 511;
		int heightBound = 423;

		unsigned short* filterCollection = new unsigned short[24];
		unsigned short* filterCollectionCount = new unsigned short[24];

		//For each row
		for (int depthArrayRowIndex = 0; depthArrayRowIndex < 424; depthArrayRowIndex++) {
			//For each pixel in the row
			for (int depthArrayColumnIndex = 0; depthArrayColumnIndex < 512; depthArrayColumnIndex++) {
				int depthIndex = depthArrayColumnIndex + (depthArrayRowIndex * 512);

				//Only 0-depth pixels can be candidate pixels
				if (depthArray[depthIndex] == 0) {
					int x = depthIndex % 512;
					int y = (depthIndex - x) / 512;


					//Zero fill the filterCollection
					for (int i = 0; i < 24; i++) {
						filterCollection[i] = 0;
					}

					int innerBandCount = 0;
					int outerBandCount = 0;

					//Search in 5x5 square around candidate pixel
					for (int yi = -2; yi < 3; yi++) {
						for (int xi = -2; xi < 3; xi++) {
							if (xi != 0 || yi != 0) {
								int xSearch = x + xi;
								int ySearch = y + yi;

								//Bounds checking current pixel
								if (xSearch >= 0 && xSearch <= widthBound &&
									ySearch >= 0 && ySearch <= heightBound) {
									int index = xSearch + (ySearch * 512);
									//We are only interested in non-zero pixels around the candidate pixel
									if (depthArray[index] != 0) {
										//Counting up frequency for each depth
										for (int i = 0; i < 24; i++) {
											if (filterCollection[i] == depthArray[index]) {
												//Depth of current pixel is already in collection, increment count
												filterCollectionCount[i]++;
												break;
											}
											else if (filterCollection[i] == 0) {
												//If we encounter a 0, we've reached the end of values already found. Start new value here.
												filterCollection[i] = depthArray[index];
												filterCollectionCount[i] = 1;
												break;
											}
										}

										//Find which band the non-0 pixel was found in and increment band counter
										if (yi != 2 && yi != -2 && xi != 2 && xi != -2) {
											innerBandCount++;
										}
										else {
											outerBandCount++;
										}
									}
								}
							}
						}
					}

					//We have inner and outer band non-zero counts. Compare against threshold to 
					//determine if candidate pixel should be changed to statistical mode of non-zero neighbors
					if (innerBandCount >= innerBandThreshold || outerBandCount >= outerBandThreshold) {
						short frequency = 0;
						short depth = 0;

						//This loop 
						for (int i = 0; i < 24; i++)
						{
							//If we find 0, we've reached the end of the collection
							if (filterCollection[i] == 0)
								break;
							if (filterCollectionCount[i] > frequency) {
								depth = filterCollection[i];
								frequency = filterCollectionCount[i];
							}
						}

						smoothDepthArray[depthIndex] = depth;
					}
				}
				else {
					smoothDepthArray[depthIndex] = depthArray[depthIndex];
				}
			}
		}

		//Clean up filterCollection arrays
		delete[] filterCollection;
		delete[] filterCollectionCount;

		return smoothDepthArray;
	}

	unsigned short* temporal_smoothing(unsigned short* current_frame, std::deque<unsigned short*> average_queue, int n_frames) {
		
		//Start by putting current frame into average_queue
		unsigned short* frame = new unsigned short[512 * 424];
		std::copy(current_frame, current_frame + (512 * 424), frame);
		average_queue.push_back(frame);

		//Return immediately if not enough frames in queue
		if (average_queue.size() < n_frames) {
			return current_frame;
		}

		//Loop through average_queue and perform moving average
		int* sumDepthArray = new int[512 * 424];
		unsigned short* averagedDepthArray = new unsigned short[512 * 424];
		int Denominator = 0;
		int Count = 1;
		for (std::deque<unsigned short*>::const_iterator it = average_queue.begin(); it < average_queue.end(); ++it) {
			unsigned short* item = *it;
			for (int y = 0; y < 424; y++) {
				for (int x = 0; x < 512; x++) {
					int index = x + (y * 512);
					sumDepthArray[index] += item[index] * Count;
				}
			}
			Denominator += Count;
			Count++;
		}

		
			for (int y = 0; y < 424; y++) {
				for (int x = 0; x < 512; x++) {
					int index = x + (y * 512);
					averagedDepthArray[index] = (unsigned short)(sumDepthArray[index] / Denominator);
				}
			}
			delete[] sumDepthArray;

		//Delete the oldest frame from queue (if more than n_frames in average_queue)
		if (average_queue.size() > n_frames) {
			unsigned short* oldest_frame = average_queue.front();
			delete[] oldest_frame;
			average_queue.pop_front();
		}


		//std::copy(averagedDepthArray, averagedDepthArray + (512 * 424), current_frame);

		return averagedDepthArray;

	}

	cv::Mat find_contours(cv::Mat depth_mat, cv::Mat drawing) {
		cv::Mat depth_img_gray, depth_img_gray_eight, canny_output;
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		depth_mat.copyTo(depth_img_gray);

		//masked_depth_norm.copyTo(depth_img_gray);
		depth_img_gray.convertTo(depth_img_gray_eight, CV_8UC1);

		//Canny edge detection on depth buffer
		cv::Canny(depth_img_gray_eight, canny_output, 100, 255 * 2, 3);

		//Draw canny output
		cv::namedWindow("Canny output");
		cv::imshow("Canny output", canny_output);

		//find contours using canny edge detection results
		cv::findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		//Approximate polygons for found contours and draw them in 8bit 3channel matrix
		drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);

		for (size_t i = 0; i < contours.size(); i++) {
			//Approximate closed contour polygons from found contours
			//approxPolyDP(contours[i], approxShape, cv::arcLength(cv::Mat(contours[i]), true)*0.04, true);  

			//Draw the polygons
			cv::drawContours(drawing, contours, i, cv::Scalar(255, 255, 255), 3);   // fill WHITE
		}
		
		cv::namedWindow("Contour output");
		cv::imshow("Contour output", drawing);
		return drawing;


	}

	






















}