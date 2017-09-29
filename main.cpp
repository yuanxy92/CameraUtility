#include "CameraUtil.h"
#include "CameraArray.h"

int main(int argc, char* argv[]) {
	CameraArray array;
	array.init();
	array.allocateBuffer(200);
	array.startRecord(12);
	array.saveCapture("E:\\Project\\CameraUtil\\data");
	array.release();
	return 0;
}