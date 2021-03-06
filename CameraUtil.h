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

// point grey camera sdk: Spinnaker SDK
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "./cuda/CameraUtilKernel.h"



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

	/*******************************************************/
	/*                 get function                        */
	/*******************************************************/
	/**
	@brief get camera numbers
	@return int: camera number
	*/
	int getCameraNum();
	/**
	@brief get serial numbers
	@return int: camera number
	*/
	std::vector<std::string> getCamSerialNums();

};

#endif