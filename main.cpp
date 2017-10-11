#include "CameraUtil.h"
#include "CameraArray.h"

#include "NPPJpegCoder.h"

int main(int argc, char* argv[]) {
	//CameraArray array;
	//array.init();
	//array.setWhiteBalance(1.10f, 1.65f);
	//array.allocateBuffer(20);
	//array.startRecord(12);
	//array.saveCapture("E:\\Project\\CameraUtil\\data");
	//array.release();

	npp::NPPJpegCoder coder;
	coder.init(1280, 800);
	unsigned char* rawdata = NULL;
	unsigned char* jpegdata = NULL;
	coder.encode(rawdata, jpegdata);

	return 0;
}