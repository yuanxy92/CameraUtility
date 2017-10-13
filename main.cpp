#include "CameraUtil.h"
#include "CameraArray.h"

#include "NPPJpegCoder.h"
#include "NPPJpegCoderKernel.h"

int main(int argc, char* argv[]) {
	CameraArray array;
	array.init();
	array.setWhiteBalance(1.10f, 1.65f);
	//array.allocateBuffer(20);
	array.allocateBufferJPEG(100);
	//array.startRecord(12);
	array.startRecordJPEG(12);
	//array.saveCapture("E:\\Project\\CameraUtil\\data");
	array.saveCaptureJPEGCompressed("E:\\Project\\CameraUtil\\data");
	array.release();

	//cv::Mat img = cv::imread("local_00.jpg");
	//cv::Mat bayerRGImg = NPPJpegCoderKernel::bgr2bayerRG(img);
	//npp::NPPJpegCoder coder;
	//coder.init(4000, 3000, 75);
	//int dataLength;
	//unsigned char* jpegdata = new unsigned char[1024 * 1024 * 10];
	//coder.encode(bayerRGImg, jpegdata, &dataLength);

	//// Write result to file.
	//std::ofstream outputFile("local_00_new.jpg", std::ios::out | std::ios::binary);
	//outputFile.write(reinterpret_cast<const char *>(jpegdata), dataLength);

	return 0;
}