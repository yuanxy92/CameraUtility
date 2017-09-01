/**
@brief C++ header file of camera utility class
@author: Shane Yuan
@date: Sep 1, 2017
*/

#ifndef __PTGREY_CAMERA_H__
#define __PTGREY_CAMERA_H__

// basic 
#include <iostream>
#include <cstdlib>
#include <fstream>

// C++ 11 parallel 
#include <thread>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// cuda
#ifdef _WIN32
#include <windows.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>

// point grey camera sdk: Spinnaker SDK
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

namespace CameraUtilKernel {

}

/**
@brief pointgrey camera utlity class
*/
class CameraUtil {
private:
	// camera array information
	unsigned int numCameras;
	Spinnaker::SystemPtr system;
	Spinnaker::CameraList camList;
	std::vector<std::string> serialnums;
public:

private:
	/**
	@brief camera start capturing
	@return int
	*/
	int startCapture();

	/**
	@brief camera stop capturing
	@return int
	*/
	int stopCapture();

public:
	CameraUtil();
	~CameraUtil();

	/**
	@brief init cameras
	@return int: 0, success; 1, failed
	*/
	int init();

	/**
	@brief stop cameras
	@return int
	*/
	int release();

	/**
	@brief read images
	@param std::vector<cv::Mat> & imgs
	@return int
	*/
	int capture(std::vector<cv::Mat> & imgs);

};

#endif