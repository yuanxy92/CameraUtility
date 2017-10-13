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
	curBufferInd = new int;
	*curBufferInd = 0;
	this->lastCapturedFrameInd = -1;
	// init compression coderes
	coders.resize(camutil.getCameraNum());
	for (size_t i = 0; i < camutil.getCameraNum(); i++) {
		coders[i].init(4000, 3000, 75);
	}
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
@brief pre-allocate buffers to cache images (JPEG compressed version)
@param std::string serialnum: serial number of reference camera
@param int frameNum: number of cached frames
@return int
*/
int CameraArray::allocateBufferJPEG(int frameNum) {
	// calculate camera buffer map
	jpegdatas.resize(frameNum);
	jpegdatalength.resize(frameNum);
	for (size_t i = 0; i < frameNum; i++) {
		jpegdatas[i].resize(camutil.getCameraNum());
		jpegdatalength[i].resize(camutil.getCameraNum());
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
@brief thread function to record image using camera
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
@brief thread function to record image using camera and compress to jpeg
*/
void camera_array_parallel_record_jpeg_(CameraUtil& util,
	std::vector< std::vector<char* > > & jpegdatas, 
	std::vector< npp::NPPJpegCoder > coders,
	int* curBufferInd,
	int frameNum,
	int fps) {
	//float time = 1000.0f / static_cast<float>(fps);
	//for (;;) {
	//	clock_t start, end;
	//	start = clock();
	//	// capture images
	//	util.capture(imgs[*curBufferInd]);
	//	*curBufferInd = (*curBufferInd + 1) % frameNum;
	//	if (*curBufferInd == 0)
	//		break;
	//	end = clock();
	//	float waitTime = time - static_cast<double>(end - start) / CLOCKS_PER_SEC * 1000;
	//	if (waitTime > 0) {
	//		std::this_thread::sleep_for(std::chrono::milliseconds((long long)waitTime));
	//	}
	//	printf("Capture one frame, sleep %f miliseconds, current buffer ind: %d ...\n",
	//		waitTime, *curBufferInd);
	//}
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

void camera_array_compress_jpeg_(npp::NPPJpegCoder coder, unsigned char* img,
	unsigned char* tempJpegdata, size_t* length, cudaStream_t stream) {
	int dataLength;
	coder.encode(img, tempJpegdata, &dataLength, stream);
	*length = dataLength;
}

/**
@brief camera start recording (JPEG compressed version)
@return int
*/
int CameraArray::startRecordJPEG(int fps) {
	float time = 1000.0f / static_cast<float>(fps);
	// init temp image buffer
	std::vector<cv::Mat> tempImg(camutil.getCameraNum());
	for (size_t i = 0; i < camutil.getCameraNum(); i++) {
		tempImg[i].create(3000, 4000, CV_8UC1);
	}
	// init temp jpeg compression data buffer
	tempJpegdata.resize(camutil.getCameraNum());
	for (size_t i = 0; i < camutil.getCameraNum(); i++) {
		tempJpegdata[i] = new unsigned char[1024 * 1024 * 10];
	}
	// init cuda stream
	std::vector<cudaStream_t > streams(camutil.getCameraNum());
	for (size_t i = 0; i < camutil.getCameraNum(); i++) {
		cudaStreamCreate(&streams[i]);
	}
	// init gpu bayer image memory
	std::vector<unsigned char*> bayer_img_ds(camutil.getCameraNum());
	for (size_t i = 0; i < camutil.getCameraNum(); i++) {
		cudaMalloc(&bayer_img_ds[i], sizeof(unsigned char) * 4000 * 3000);
	}
	// start recording
	for (;;) {
		clock_t start, end;
		start = clock();
		// capture images
		camutil.capture(tempImg);
		// copy data to gpu
		for (size_t i = 0; i < camutil.getCameraNum(); i++) {
			cudaMemcpyAsync(bayer_img_ds[i], tempImg[i].data, 
				sizeof(unsigned char) * 4000 * 3000, 
				cudaMemcpyHostToDevice, streams[i]);			
		}
		// compression
		for (size_t i = 0; i < camutil.getCameraNum(); i++) {
			// synchronization
			cudaStreamSynchronize(streams[i]);
			camera_array_compress_jpeg_(
				coders[i], bayer_img_ds[i], tempJpegdata[i],
				&jpegdatalength[*curBufferInd][i], streams[i]);
		}
		for (size_t i = 0; i < camutil.getCameraNum(); i++) {
			// synchronization
			cudaStreamSynchronize(streams[i]);
			jpegdatas[*curBufferInd][i] = new char[jpegdatalength[*curBufferInd][i]];
			memcpy(jpegdatas[*curBufferInd][i], tempJpegdata[i], jpegdatalength[*curBufferInd][i]);
		}
		// increase buffer index
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
@brief preview capture
*/
int CameraArray::saveCaptureJPEGCompressed(std::string dir) {
	for (size_t j = 0; j < jpegdatas[0].size(); j++) {
		for (size_t i = 0; i < jpegdatas.size(); i++) {
			std::string name = cv::format("%s/img_%02d_%05d.jpg", dir.c_str(), j, i);
			std::ofstream outputFile(name.c_str(), std::ios::out | std::ios::binary);
			outputFile.write(jpegdatas[i][j], jpegdatalength[i][j]);
		}
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

/**
@brief set white balance
@param int ind: input index of camera
@param float red: red value in white balance
@param float blue: blue value in white balance
@return int
*/
int CameraArray::setWhiteBalance(int ind, float red, float blue) {
	this->camutil.setWhiteBalance(ind, red, blue);
	return 0;
}

/**
@brief set white balance for all the cameras
@param float red: red value in white balance
@param float blue: blue value in white balance
@return int
*/
int CameraArray::setWhiteBalance(float red, float blue) {
	this->camutil.setWhiteBalance(red, blue);
	return 0;
}




