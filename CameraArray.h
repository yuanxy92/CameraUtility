/**
@brief C++ header file of camera array class
@author: Shane Yuan 
@date: Sep 2, 2017
*/

#ifndef __GIGA_RENDER_CAMERA_ARRAY__
#define __GIGA_RENDER_CAMERA_ARRAY__


// camera utility
#include "CameraUtil.h"

#include "NPPJpegCoder.h"

/**************************************************************************/
/*          VideoIO class, read video frames from camera array            */
/**************************************************************************/
class CameraArray {
private:
	CameraUtil camutil;
public:
	int* curBufferInd;
	int lastCapturedFrameInd;
	int fps;
	int frameNum;
	std::vector< std::vector<cv::Mat> > bufferImgs;
	std::thread th;

	// jpeg compression
	// npp class for compression
	std::vector<npp::NPPJpegCoder> coders;
	std::vector<unsigned char*> tempJpegdata;
	std::vector<std::vector<char*>> jpegdatas;
	std::vector<std::vector<size_t>> jpegdatalength;
private:
	/**
	@brief write recorded video into file
	@param std::string dir: dir to save recorded videos
	@return int
	*/
	int writeVideo(std::string dir);

public:
	CameraArray();
	~CameraArray();

	/**
	@brief init camera array class
	*/
	int init();

	/**
	@brief pre-allocate buffers to cache images
	@param std::string serialnum: serial number of reference camera
	@param int frameNum: number of cached frames
	@return int
	*/
	int allocateBuffer(int frameNum);

	/**
	@brief pre-allocate buffers to cache images (JPEG compressed version)
	@param std::string serialnum: serial number of reference camera
	@param int frameNum: number of cached frames
	@return int
	*/
	int allocateBufferJPEG(int frameNum);

	/**
	@brief start capture
	@param int fps;
	@return int
	*/
	int startCapture(int fps);

	/**
	@brief camera start recording
	@return int
	*/
	int startRecord(int fps);

	/**
	@brief camera start recording (JPEG compressed version)
	@return int
	*/
	int startRecordJPEG(int fps);

	/**
	@brief preview capture
	*/
	int saveCapture(std::string dir);

	/**
	@brief preview capture
	*/
	int saveCaptureJPEGCompressed(std::string dir);

	/**
	@brief stop capture
	@return int
	*/
	int stopCapture();

	/**
	@brief release camera array
	*/
	int release();

	/**
	@brief capture bayer image to update
	@param std::vector<cv::Mat>& bayerImgs: output bayer images
	@return int
	*/
	int captureOneFrameBayer(std::vector<cv::Mat>& bayerImgs);

	/**
	@brief capture next frame bayer image
	@param std::vector<cv::Mat>& bayerImgs: output bayer images
	@return int
	*/
	int captureNextFrame(std::vector<cv::Mat>& bayerImgs);

	/**
	@brief set white balance
	@param int ind: input index of camera
	@param float red: red value in white balance
	@param float blue: blue value in white balance
	@return int
	*/
	int setWhiteBalance(int ind, float red, float blue);

	/**
	@brief set white balance for all the cameras
	@param float red: red value in white balance
	@param float blue: blue value in white balance
	@return int
	*/
	int setWhiteBalance(float red, float blue);

};

#endif