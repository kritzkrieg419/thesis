#include "Grabber.h"

#include <kinect.h>
#include <opencv2/imgproc/imgproc.hpp> // cv color
#include <opencv2/highgui/highgui.hpp>
#include <boost/thread.hpp>

namespace Grabber {

	HRESULT hResult;
	IKinectSensor* pSensor;
	ICoordinateMapper* pCoordinateMapper;
	IColorFrameSource* pColorSource;
	IColorFrameReader* pColorReader;
	IDepthFrameSource* pDepthSource;
	IDepthFrameReader* pDepthReader;
	IInfraredFrameSource* pInfraredSource;
	IInfraredFrameReader* pInfraredReader;
	IBodyFrameSource* pBodyFrameSource;
	IBodyIndexFrameSource* pBodyIndexFrameSource;

	IBodyFrameReference* pBodyFrameReference;
	IBodyIndexFrameReference* pBodyIndexFrameReference;
	IBodyFrameReader* pBodyFrameReader;
	IBodyIndexFrameReader* pBodyIndexFrameReader;

	CameraSpacePoint headPos;
	CameraSpacePoint leftHandPos;
	CameraSpacePoint leftThumbPos;
	CameraSpacePoint leftWristPos;
	CameraSpacePoint leftHandTipPos;
	unsigned char* bodyIndexBuffer;
	bool tracking_body = 0;

	int colorWidth;
	int colorHeight;
	unsigned int colorBuffer2Size;
	unsigned char* colorBuffer2;
	std::vector<RGBQUAD> colorBuffer;

	int depthWidth;
	int depthHeight;
	unsigned short* depthBuffer;

	int infraredWidth;
	int infraredHeight;
	int bodyIndexBufferLength;
	std::vector<UINT16> infraredBuffer;

	cv::Mat img;
	cv::Mat depth_img_rgb_space;
	cv::Mat depth_img_rgb_space_norm;
	cv::Mat color_img;
	cv::Mat depth_img_gray;

	double min;
	double max;

	boost::mutex mtx;

	template <class Interface> inline void safe_release(Interface *& ppT)
	{
		if (ppT)
		{
			ppT->Release();
			ppT = NULL;
		}
	}

	void release_all()
	{
		// End Processing
		if (pSensor) {
			pSensor->Close();
		}
		safe_release(pSensor);
		safe_release(pCoordinateMapper);
		safe_release(pColorSource);
		safe_release(pColorReader);
		safe_release(pDepthSource);
		safe_release(pDepthReader);
		safe_release(pInfraredSource);
		safe_release(pInfraredReader);
		safe_release(pBodyFrameSource);
		safe_release(pBodyFrameReader);
		safe_release(pBodyIndexFrameSource);
		safe_release(pBodyIndexFrameReader);
	}

	void grabber_init()
	{

		// Create sensor instance
		hResult = GetDefaultKinectSensor(&pSensor);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Error: GetDefaultSensor");
		}

		// Open the sensor
		hResult = pSensor->Open();
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Error: IKinectSensor::Open()");
		}

		// Coordinate Mapper
		hResult = pSensor->get_CoordinateMapper(&pCoordinateMapper);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Error: IKinectSensor::get_CoordinateMapper()");
		}

		// Color Frame Source
		hResult = pSensor->get_ColorFrameSource(&pColorSource);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IKinectSensor::get_ColorFrameSource()");
		}

		// Depth Frame Source
		hResult = pSensor->get_DepthFrameSource(&pDepthSource);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IKinectSensor::get_DepthFrameSource()");
		}

		// Infrared Frame Source
		hResult = pSensor->get_InfraredFrameSource(&pInfraredSource);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IKinectSensor::get_InfraredFrameSource()");
		}

		//Body Frame Source
		hResult = pSensor->get_BodyFrameSource(&pBodyFrameSource);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IKinectSensor::get_BodyFrameSource()");
		}

		//Body Index Frame Source
		hResult = pSensor->get_BodyIndexFrameSource(&pBodyIndexFrameSource);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IKinectSensor::get_BodyIndexFrameSource()");
		}



		/////////////////////////////////////////////////////////////////////////////
		// Color Frame Size
		IFrameDescription* pColorDescription;
		hResult = pColorSource->get_FrameDescription(&pColorDescription);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IColorFrameSource::get_FrameDescription()");
		}

		hResult = pColorDescription->get_Width(&colorWidth); // 1920
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IFrameDescription::get_Width()");
		}

		hResult = pColorDescription->get_Height(&colorHeight); // 1080
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IFrameDescription::get_Height()");
		}

		safe_release(pColorDescription);

		// To Reserve Color Frame Buffer
		//colorBuffer = new unsigned char[colorWidth*colorHeight * 4 * sizeof(unsigned char)];
		colorBuffer.resize(colorWidth * colorHeight);

		colorBuffer2Size = colorWidth*colorHeight * 4 * sizeof(unsigned char);
		colorBuffer2 = new unsigned char[colorBuffer2Size];
		/////////////////////////////////////////////////////////////////////////////
		// Depth Frame Size
		IFrameDescription* pDepthDescription;
		hResult = pDepthSource->get_FrameDescription(&pDepthDescription);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IDepthFrameSource::get_FrameDescription()");
		}

		hResult = pDepthDescription->get_Width(&depthWidth); // 512
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IFrameDescription::get_Width()");
		}

		hResult = pDepthDescription->get_Height(&depthHeight); // 424
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IFrameDescription::get_Height()");
		}

		safe_release(pDepthDescription);

		// To Reserve Depth Frame Buffer
		depthBuffer = new unsigned short[depthWidth * depthHeight];
		//depthBuffer.resize(depthWidth * depthHeight);

		/////////////////////////////////////////////////////////////////////////////
		// Retrieved Infrared Frame Size
		IFrameDescription* pInfraredDescription;
		hResult = pInfraredSource->get_FrameDescription(&pInfraredDescription);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IInfraredFrameSource::get_FrameDescription()");
		}

		hResult = pInfraredDescription->get_Width(&infraredWidth); // 512
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IFrameDescription::get_Width()");
		}

		hResult = pInfraredDescription->get_Height(&infraredHeight); // 424
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IFrameDescription::get_Height()");
		}

		safe_release(pInfraredDescription);

		// To Reserve Infrared Frame Buffer
		infraredBuffer.resize(infraredWidth * infraredHeight);
	}

	void grabber_start()
	{
		// Open Color Frame Reader
		hResult = pColorSource->OpenReader(&pColorReader);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IColorFrameSource::OpenReader()");
		}

		// Open Depth Frame Reader
		hResult = pDepthSource->OpenReader(&pDepthReader);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IDepthFrameSource::OpenReader()");
		}

		// Open Infrared Frame Reader
		hResult = pInfraredSource->OpenReader(&pInfraredReader);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IInfraredFrameSource::OpenReader()");
		}

		//Open Body Frame Reader
		hResult = pBodyFrameSource->OpenReader(&pBodyFrameReader);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IBodyFrameSource::OpenReader()");
		}

		//Open Body Index Frame Reader
		hResult = pBodyIndexFrameSource->OpenReader(&pBodyIndexFrameReader);
		if (FAILED(hResult)) {
			release_all();
			throw std::exception("Exception : IBodyIndexFrameSource::OpenReader()");
		}
	}

	void frame_acquisition()
	{
		boost::mutex::scoped_lock(mtx);

		////
		// Acquire Latest Color Frame
		IColorFrame* pColorFrame = nullptr;
		hResult = pColorReader->AcquireLatestFrame(&pColorFrame);
		if (SUCCEEDED(hResult)) {
			// Retrieved Color Data
			
			HRESULT hResult2 = pColorFrame->CopyConvertedFrameDataToArray(colorBuffer2Size, colorBuffer2, ColorImageFormat::ColorImageFormat_Bgra);
			
			hResult = pColorFrame->CopyConvertedFrameDataToArray(colorBuffer.size() * sizeof(RGBQUAD), reinterpret_cast<BYTE*>(&colorBuffer[0]), ColorImageFormat::ColorImageFormat_Bgra);
			
			if (FAILED(hResult) || FAILED(hResult2)) {
				release_all();
				throw std::exception("Exception : IColorFrame::CopyConvertedFrameDataToArray()");
			}
		}
		safe_release(pColorFrame);

		// Acquire Latest Depth Frame
		IDepthFrame* pDepthFrame = nullptr;
		hResult = pDepthReader->AcquireLatestFrame(&pDepthFrame);
		if (SUCCEEDED(hResult)) {
			// Retrieved Depth Data
			hResult = pDepthFrame->CopyFrameDataToArray(depthWidth*depthHeight, depthBuffer);
			//depthBuffer = smoothDepthArray(depthBuffer, 2, 5);
			if (FAILED(hResult)) {
				release_all();
				throw std::exception("Exception : IDepthFrame::CopyFrameDataToArray()");
			}
		}
		safe_release(pDepthFrame);

		// Acquire Latest Infrared Frame
		IInfraredFrame* pInfraredFrame = nullptr;
		hResult = pInfraredReader->AcquireLatestFrame(&pInfraredFrame);
		if (SUCCEEDED(hResult)) {
			// Retrieved Infrared Data
			hResult = pInfraredFrame->CopyFrameDataToArray(infraredBuffer.size(), &infraredBuffer[0]);
			if (FAILED(hResult)) {
				release_all();
				throw std::exception("Exception : IInfraredFrame::CopyFrameDataToArray()");
			}
		}
		safe_release(pInfraredFrame);

		//Acquire Body Frame
		IBodyFrame* pBodyFrame = nullptr;
		hResult = pBodyFrameReader->AcquireLatestFrame(&pBodyFrame);
		if (SUCCEEDED(hResult)) {
			//Retrieved body frame
			IBody *bodies[BODY_COUNT] = { 0 };
			hResult = pBodyFrame->GetAndRefreshBodyData(_countof(bodies), bodies);
			if (SUCCEEDED(hResult)) {
				processBodies(BODY_COUNT, bodies);
				for (unsigned int bodyIndex = 0; bodyIndex < _countof(bodies); bodyIndex++) {
					safe_release(bodies[bodyIndex]);
				}
			}
		}
		safe_release(pBodyFrame);

		//Acquire Body Index Frame
		IBodyIndexFrame* pBodyIndexFrame = nullptr;
		hResult = pBodyIndexFrameReader->AcquireLatestFrame(&pBodyIndexFrame);
		if (SUCCEEDED(hResult)) {
			//Retrieved Body Index frame
			IFrameDescription* bodyIndexFrameDescription;
			pBodyIndexFrame->get_FrameDescription(&bodyIndexFrameDescription);
			int indexHeight;
			int indexWidth;
			bodyIndexFrameDescription->get_Height(&indexHeight);
			bodyIndexFrameDescription->get_Width(&indexWidth);
			bodyIndexBuffer = new unsigned char[indexHeight*indexWidth];
			bodyIndexBufferLength = indexHeight*indexWidth;
			pBodyIndexFrame->CopyFrameDataToArray(indexHeight*indexWidth, bodyIndexBuffer);
		}
		safe_release(pBodyIndexFrame);

		////////////////////////////////////////////////////////////////////////////
		color_img = cv::Mat(cv::Size(colorWidth, colorHeight), CV_8UC4, colorBuffer2);

		// Depth calc
		// Create Mat object from depth array
		img = cv::Mat(cv::Size(depthWidth, depthHeight), CV_16UC1, depthBuffer);

		// convert grayscale to color image
		depth_img_gray = img;
		cv::cvtColor(img, depth_img_rgb_space, cv::COLOR_GRAY2BGR, 3);

		//Normalize values in depth buffer array for better visualization
		cv::minMaxIdx(depth_img_rgb_space, &min, &max);
		cv::convertScaleAbs(depth_img_rgb_space, depth_img_rgb_space_norm, 65535 / max);

		////
	}
	///////////// HELPER FUNCTIONS ////////////////////
	void processBodies(const unsigned int &bodyCount, IBody **bodies) {
		for (unsigned int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++) {
			IBody *body = bodies[bodyIndex];

			//Get tracking status for current body
			BOOLEAN isTracked = false;
			HRESULT hr = body->get_IsTracked(&isTracked);
			if (FAILED(hr) || isTracked == false) {
				//Skip this body
				continue;
			}

			//Get joint properties for skeleton
			Joint joints[JointType_Count];
			hr = body->GetJoints(_countof(joints), joints);
			if (SUCCEEDED(hr)) {
				//We have joint properties
				headPos = joints[JointType_Head].Position;
				leftHandPos = joints[JointType_HandLeft].Position;
				leftThumbPos = joints[JointType_ThumbLeft].Position;
				leftWristPos = joints[JointType_WristLeft].Position;
				leftHandTipPos = joints[JointType_HandTipLeft].Position;
				if (joints[JointType_HandLeft].TrackingState == 2) {
					tracking_body = 1;
				}
				else {
					tracking_body = 0;
				}
			}
		}
	}

	/*
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertRGBDepthToPointXYZRGB()
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

		cloud->width = static_cast<uint32_t>(depthWidth);
		cloud->height = static_cast<uint32_t>(depthHeight);
		cloud->is_dense = false;

		cloud->points.resize(cloud->height * cloud->width);

		pcl::PointXYZRGB* pt = &cloud->points[0];
		for (int y = 0; y < depthHeight; y++) {
			for (int x = 0; x < depthWidth; x++, pt++) {
				pcl::PointXYZRGB point;

				DepthSpacePoint depthSpacePoint = { static_cast<float>(x), static_cast<float>(y) };
				UINT16 depth = depthBuffer[y * depthWidth + x];

				// Coordinate Mapping Depth to Color Space, and Setting PointCloud RGB
				ColorSpacePoint colorSpacePoint = { 0.0f, 0.0f };
				pCoordinateMapper->MapDepthPointToColorSpace(depthSpacePoint, depth, &colorSpacePoint);
				int colorX = static_cast<int>(std::floor(colorSpacePoint.X + 0.5f));
				int colorY = static_cast<int>(std::floor(colorSpacePoint.Y + 0.5f));
				if ((0 <= colorX) && (colorX < colorWidth) && (0 <= colorY) && (colorY < colorHeight)) {
					RGBQUAD color = colorBuffer[colorY * colorWidth + colorX];
					point.b = color.rgbBlue;
					point.g = color.rgbGreen;
					point.r = color.rgbRed;
				}

				// Coordinate Mapping Depth to Camera Space, and Setting PointCloud XYZ
				CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
				pCoordinateMapper->MapDepthPointToCameraSpace(depthSpacePoint, depth, &cameraSpacePoint);
				if ((0 <= colorX) && (colorX < colorWidth) && (0 <= colorY) && (colorY < colorHeight)) {
					point.x = cameraSpacePoint.X;
					point.y = cameraSpacePoint.Y;
					point.z = cameraSpacePoint.Z;
				}

				*pt = point;
			}
		}

		return cloud;
	}
	*/
	unsigned short* smoothDepthArray(unsigned short* depthArray, int innerBandThreshold, int outerBandThreshold) {
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
									int index = xSearch + (ySearch*512);
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



	unsigned short* get_p_depth_buffer()
	{
		boost::mutex::scoped_lock(mtx);
		return depthBuffer;
	}

	ICoordinateMapper* get_pCoordinateMapper()
	{
		return pCoordinateMapper;
	}

	cv::Mat get_depth_norm_mat()
	{
		//Get current depth buffer
		cv::Mat temp = cv::Mat(cv::Size(depthWidth, depthHeight), CV_16UC1, depthBuffer);

		// convert grayscale to color image
		cv::cvtColor(temp, depth_img_rgb_space, cv::COLOR_GRAY2BGR, 3);

		//Normalize values in depth buffer array for better visualization
		double min;
		double max;
		cv::minMaxIdx(depth_img_rgb_space, &min, &max);
		cv::convertScaleAbs(depth_img_rgb_space, depth_img_rgb_space_norm, 65535 / max);
		return depth_img_rgb_space_norm;
	}

	cv::Mat get_depth_mat() {
		return depth_img_gray;
	}

	cv::Mat get_color_mat()
	{
		return color_img;
	}

	ColorSpacePoint getLeftHandTip()
	{
		ColorSpacePoint retval;
		pCoordinateMapper->MapCameraPointToColorSpace(leftHandTipPos, &retval);
		return retval;
	}

	ColorSpacePoint getLeftHandWrist()
	{
		ColorSpacePoint retval;
		pCoordinateMapper->MapCameraPointToColorSpace(leftWristPos, &retval);
		return retval;
	}

	unsigned short* getDepthBuffer() {
		return depthBuffer;
	}

	std::vector<DepthSpacePoint> getColorToDepthMapping() {
		std::vector<DepthSpacePoint> depthSpace(1920 * 1080);
		HRESULT hr = pCoordinateMapper->MapColorFrameToDepthSpace(512 * 424, depthBuffer, 1920 * 1080, &depthSpace[0]);
		return depthSpace;
	}

	ColorSpacePoint getLeftHandCentroid()
	{
		ColorSpacePoint retval;
		pCoordinateMapper->MapCameraPointToColorSpace(leftHandPos, &retval);
		return retval;
	}

	ColorSpacePoint getLeftHandThumb() {
		ColorSpacePoint retval;
		pCoordinateMapper->MapCameraPointToColorSpace(leftThumbPos, &retval);
		return retval;
	}

	unsigned char* getBodyIndexBuffer() {
		return bodyIndexBuffer;
	}

	int getBodyIndexBufferLength() {
		return bodyIndexBufferLength;
	}

	bool isTrackingBody() {
		return tracking_body;
	}
}
