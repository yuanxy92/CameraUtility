/**
@brief C++ source file of camera utility class
@author: Shane Yuan
@date: Sep 1, 2017
*/

#include "CameraUtil.h"

CameraUtil::CameraUtil(){}
CameraUtil::~CameraUtil(){}

/**
@brief camera start capturing
@return int
*/
int CameraUtil::startCapture() {
	int result = 0;
	Spinnaker::CameraPtr pCam = NULL;
	try {
		for (int i = 0; i < camList.GetSize(); i++) {
			// Select camera
			pCam = camList.GetByIndex(i);
			// Set acquisition mode to continuous
			Spinnaker::GenApi::CEnumerationPtr ptrAcquisitionMode = pCam->GetNodeMap().GetNode("AcquisitionMode");
			if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode)) {
				std::cout << "Unable to set acquisition mode to continuous (node retrieval; camera " << i << "). Aborting..." << std::endl << std::endl;
				return -1;
			}
			Spinnaker::GenApi::CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
			if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous)) {
				std::cout << "Unable to set acquisition mode to continuous (entry 'continuous' retrieval " << i << "). Aborting..." << std::endl << std::endl;
				return -1;
			}
			int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();
			ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);
			std::cout << "Camera " << i << " acquisition mode set to continuous..." << std::endl;
			// Begin acquiring images
			pCam->BeginAcquisition();
			std::cout << "Camera " << i << " started acquiring images..." << std::endl;
			// Retrieve device serial number for filename
			serialnums[i] = "";
			Spinnaker::GenApi::CStringPtr ptrStringSerial = pCam->GetTLDeviceNodeMap().GetNode("DeviceSerialNumber");
			if (IsAvailable(ptrStringSerial) && IsReadable(ptrStringSerial)) {
				serialnums[i] = ptrStringSerial->GetValue();
				std::cout << "Camera " << i << " serial number set to " << serialnums[i] << "..." << std::endl;
			}
			std::cout << std::endl;
		}
	}
	catch (Spinnaker::Exception &e) {
		std::cout << "Error: " << e.what() << std::endl;
		result = -1;
	}
	return result;
}

/**
@brief camera stop capturing
@return int
*/
int CameraUtil::stopCapture() {
	int result = 0;
	Spinnaker::CameraPtr pCam = NULL;
	try {
		for (int i = 0; i < camList.GetSize(); i++) {
			// Select camera
			pCam = camList.GetByIndex(i);
			pCam->EndAcquisition();
		}
	}
	catch (Spinnaker::Exception &e) {
		std::cout << "Error: " << e.what() << std::endl;
		result = -1;
	}
	return result;
}


/**
@brief init cameras
@return int
*/
int CameraUtil::init() {
	// get camera lists
	system = Spinnaker::System::GetInstance();
	camList = system->GetCameras();
	numCameras = camList.GetSize();
	serialnums.resize(numCameras);
	// printf information
	std::cout << cv::format("Number of cameras detected: %d :", numCameras) << std::endl;
	for (size_t i = 0; i < numCameras; i++) {
		Spinnaker::GenApi::INodeMap & nodeMap = camList.GetByIndex(i)->GetTLDeviceNodeMap();
		std::cout << cv::format("\ncamera %d:\n", i);
		try {
			Spinnaker::GenApi::FeatureList_t features;
			Spinnaker::GenApi::CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
			if (IsAvailable(category) && IsReadable(category)) {
				category->GetFeatures(features);
				Spinnaker::GenApi::FeatureList_t::const_iterator it;
				for (it = features.begin(); it != features.end(); ++it) {
					Spinnaker::GenApi::CNodePtr pfeatureNode = *it;
					std::cout << pfeatureNode->GetName() << " : ";
					Spinnaker::GenApi::CValuePtr pValue = (Spinnaker::GenApi::CValuePtr)pfeatureNode;
					std::cout << (IsReadable(pValue) ? pValue->ToString() : "Node not readable");
					std::cout << std::endl;
				}
			}
			else {
				std::cout << "Device control information not available." << std::endl;
			}
		}
		catch (Spinnaker::Exception &e) {
			std::cout << "Error: " << e.what() << std::endl;
			return -1;
		}
	}
	std::cout << std::endl;
	// initialize all the cameras
	for (size_t i = 0; i < numCameras; i++) {
		try {
			// Select camera
			Spinnaker::CameraPtr pCam = camList.GetByIndex(i);
			// Initialize camera
			pCam->Init();
		}
		catch (Spinnaker::Exception &e) {
			std::cout << "Error: " << e.what() << std::endl;
			return 0;
		}
	}
	// start capture
	this->startCapture();
	return 0;
}

/**
@brief init cameras
@return int
*/
int CameraUtil::release() {
	// stop capture
	this->stopCapture();
	// de-initialize all the cameras
	for (size_t i = 0; i < numCameras; i++) {
		try {
			// Select camera
			Spinnaker::CameraPtr pCam = camList.GetByIndex(i);
			// Initialize camera
			pCam->DeInit();
		}
		catch (Spinnaker::Exception &e) {
			std::cout << "Error: " << e.what() << std::endl;
			return 0;
		}
	}
	// clear variable
	camList.Clear();
	system->ReleaseInstance();
	return 0;
}

/**
@brief parallel capture function
*/
void parallel_capture_(Spinnaker::CameraPtr & pCam, cv::Mat & img) {
	// capture
	Spinnaker::ImagePtr image = pCam->GetNextImage();
	void* data = image->GetData();
	std::memcpy(img.data, data, sizeof(unsigned char) * img.rows * img.cols);
}

/**
@brief read images
@param std::vector<cv::Mat> imgs
@return int
*/
int CameraUtil::capture(std::vector<cv::Mat> & imgs) {
	std::vector<std::thread> ths;
	// start thread
	for (size_t i = 0; i < numCameras; i++) {
		ths.push_back(std::thread(parallel_capture_, camList.GetByIndex(i), imgs[i]));
	}
	// stop thread
	for (size_t i = 0; i < numCameras; i++) {
		ths[i].join();
	}
	return 0;
}

