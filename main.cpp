#include "CameraUtil.h"

int main(int argc, char* argv[]) {
	std::vector<cv::Mat> imgs(2);
	imgs[0].create(3000, 4000, CV_8U);
	imgs[1].create(3000, 4000, CV_8U);
	CameraUtil util;
	util.init();
	util.capture(imgs);
	util.release();

	cv::Mat img1 = CameraUtilKernel::demosaic(imgs[0]);
	cv::Mat img2 = CameraUtilKernel::demosaic(imgs[1]);

	return 0;
}