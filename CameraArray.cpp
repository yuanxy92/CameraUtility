/**
@brief C++ source file of camera array class
@author: Shane Yuan
@date: Sep 2, 2017
*/

#include "CameraArray.h"

/**************************************************************************/
/*          VideoIO class, read video frames from camera array            */
/**************************************************************************/
CameraArray::CameraArray() {}
CameraArray::~CameraArray() {}

/**
@brief init camera array class
*/
int CameraArray::init() {
	camutil.init();
	camutil.setWhiteBalance(1.10f, 1.60f);
	curBufferInd = new int;
	*curBufferInd = 0;
	this->lastCapturedFrameInd = -1;
	return 0;
}

/**
@brief release camera array
*/
int CameraArray::release() {
	camutil.release();
	delete curBufferInd;
	return 0;
}


/**
@brief pre-allocate buffers to cache images
@param std::string serialnum: serial number of reference camera
@param int frameNum: number of cached frames
@return int
*/
int CameraArray::allocateBuffer(int frameNum) {
	// calculate camera buffer map
	bufferImgs.resize(frameNum);
	for (size_t i = 0; i < frameNum; i++) {
		bufferImgs[i].resize(camutil.getCameraNum());
		for (size_t j = 0; j < camutil.getCameraNum(); j++) {
			bufferImgs[i][j].create(3000, 4000, CV_8U);
		}
	}
	this->frameNum = frameNum;
	return 0;
}

/**
@brief thread function to capture image using camera
*/
void camera_array_parallel_capture_(CameraUtil& util,
	std::vector< std::vector<cv::Mat> > & imgs, int* curBufferInd, int fps) {
	int frameNum = imgs.size();
	float time = 1000.0f / static_cast<float>(fps);
	for (;;) {
		clock_t start, end;
		start = clock();
		// capture images
		util.capture(imgs[*curBufferInd]);
		*curBufferInd = (*curBufferInd + 1) % frameNum;
		end = clock();
		float waitTime = time - static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000;
		if (waitTime > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds((long long)waitTime));
		}
		printf("Capture one frame, sleep %f miliseconds, current buffer ind: %d ...\n",
			waitTime, *curBufferInd);
	}
}

/**
@brief thread function to capture image using camera
*/
void camera_array_parallel_record_(CameraUtil& util,
	std::vector< std::vector<cv::Mat> > & imgs, int* curBufferInd, int fps) {
	int frameNum = imgs.size();
	float time = 1000.0f / static_cast<float>(fps);
	for (;;) {
		clock_t start, end;
		start = clock();
		// capture images
		util.capture(imgs[*curBufferInd]);
		*curBufferInd = (*curBufferInd + 1) % frameNum;
		if (*curBufferInd == 0)
			break;
		end = clock();
		float waitTime = time - static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000;
		if (waitTime > 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds((long long)waitTime));
		}
		printf("Capture one frame, sleep %f miliseconds, current buffer ind: %d ...\n",
			waitTime, *curBufferInd);
	}
}

/**
@brief start capture
@param int fps;
@return int
*/
int CameraArray::startCapture(int fps) {
	this->fps = fps;
	th = std::thread(camera_array_parallel_capture_, std::ref(camutil), std::ref(bufferImgs), curBufferInd, fps);
	return 0;
}

/**
@brief write recorded video into file
@param std::string dir: dir to save recorded videos
@return int
*/
int CameraArray::writeVideo(std::string dir) {

	return 0;
}

/**
@brief camera start recording
@return int
*/
int CameraArray::startRecord(int fps) {
	this->fps = fps;
	th = std::thread(camera_array_parallel_record_, std::ref(camutil), std::ref(bufferImgs), curBufferInd, fps);
	th.join();


	return 0;
}


/**
@brief preview capture
*/
int CameraArray::saveCapture(std::string dir) {
	for (size_t j = 0; j < bufferImgs[0].size(); j++) {
		cv::VideoWriter writer(cv::format("%s/cam_%02d.avi", dir.c_str(), j), 
			cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 12, cv::Size(4000, 3000), true);
		for (size_t i = 0; i < bufferImgs.size(); i++) {
			cv::Mat imgcolor = CameraUtilKernel::demosaic(bufferImgs[i][j]);
			//cv::imwrite(cv::format("%s/cam_%02d_frame_%05d.jpg", dir.c_str(), j, i), imgcolor);
			writer << imgcolor;
		}
		writer.release();
	}
	return 0;
}

/**
@brief stop capture
@return int
*/
int CameraArray::stopCapture() {
	th.join();
	return 0;
}





