#include "CameraUtil.h"
#include "CameraArray.h"

int main(int argc, char* argv[]) {
	CameraArray array;
	array.init();
	array.allocateBuffer(20);
	array.startCapture(12);
	array.previewCapture();
	array.release();
	return 0;
}