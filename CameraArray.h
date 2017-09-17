/**
@brief C++ header file of camera array class
@author: Shane Yuan 
@date: Sep 2, 2017
*/

#ifndef __GIGA_RENDER_CAMERA_ARRAY__
#define __GIGA_RENDER_CAMERA_ARRAY__


// camera utility
#include "CameraUtil.h"

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
	@brief preview capture
	*/
	int previewCapture();

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

};

#endif