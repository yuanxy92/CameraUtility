#include "CameraUtil.h"
#include "CameraArray.h"

#include "NPPJpegCoder.h"
#include "NPPJpegCoderKernel.h"

#include <time.h>

#define MEASURE_KERNEL_TIME

int main(int argc, char* argv[]) {
	CameraArray array;
	array.init();
	array.setWhiteBalance(1.10f, 1.65f);
	//array.allocateBuffer(20);
	array.allocateBufferJPEG(2000);
	//array.startRecord(12);
	array.startRecordJPEG(12);
	//array.saveCapture("E:\\Project\\CameraUtil\\data");
	array.saveCaptureJPEGCompressed("E:\\Project\\CameraUtil\\data");
	array.release();

	return 0;

	size_t cameraNum = 8;

	std::vector<cv::Mat> bayerimgs(cameraNum);
	std::vector<unsigned char*> imgs(cameraNum);
	for (size_t i = 0; i < cameraNum; i++) {
		cv::Mat img = cv::imread(cv::format("local_%02d.jpg", i));
		bayerimgs[i] = NPPJpegCoderKernel::bgr2bayerRG(img);
		cudaMalloc(&imgs[i], sizeof(unsigned char*) * 4000 * 3000);
	}

	std::vector<npp::NPPJpegCoder> coders(cameraNum);
	for (size_t i = 0; i < cameraNum; i++) {
		coders[i].init(4000, 3000, 75);
	}

	std::vector<unsigned char*> tempJpegdatas(cameraNum);
	std::vector<char*> jpegdatas(cameraNum);
	std::vector<int> dataLengths(cameraNum);
	std::vector<cudaStream_t> streams(cameraNum);

	for (size_t i = 0; i < cameraNum; i++) {
		cudaStreamCreate(&streams[i]);
		tempJpegdatas[i] = new unsigned char[1024 * 1024 * 10];
	}


	clock_t start, end;
	start = clock();

	// upload
	for (size_t i = 0; i < cameraNum; i++) {
		cudaMemcpyAsync(imgs[i], bayerimgs[i].data, sizeof(unsigned char) * 4000 * 3000,
			cudaMemcpyHostToDevice, streams[i]);
	}

	// compression
#ifdef MEASURE_KERNEL_TIME
	cudaEvent_t cudastart, cudastop;
	float elapsedTime;
	cudaEventCreate(&cudastart);
	cudaEventRecord(cudastart, 0);
#endif

	for (size_t i = 0; i < cameraNum; i++) {
		// synchronization
		cudaStreamSynchronize(streams[i]);
		int dataLength;
		coders[i].encode(imgs[i], tempJpegdatas[i], &dataLength, streams[i]);
		dataLengths[i] = dataLength;
	}

	for (size_t i = 0; i < cameraNum; i++) {
		// synchronization
		cudaStreamSynchronize(streams[i]);
	}

#ifdef MEASURE_KERNEL_TIME
	cudaEventCreate(&cudastop);
	cudaEventRecord(cudastop, 0);
	cudaEventSynchronize(cudastop);
	cudaEventElapsedTime(&elapsedTime, cudastart, cudastop);
	printf("JPEG encode: (file:%s, line:%d) elapsed time : %f ms\n", __FILE__, __LINE__, elapsedTime);
#endif

	for (size_t i = 0; i < cameraNum; i++) {
		// synchronization
		jpegdatas[i] = new char[dataLengths[i]];
		memcpy(jpegdatas[i], tempJpegdatas[i], dataLengths[i] * sizeof(unsigned char));
	}

	end = clock();
	float waitTime = (double)(end - start) / CLOCKS_PER_SEC * 1000;
	printf("Compress one frame, cost %f miliseconds ...\n", waitTime);

	for (size_t i = 0; i < cameraNum; i++) {
		std::string name = cv::format("compresse_%02d.jpg", i);
		std::ofstream outputFile(name.c_str(), std::ios::out | std::ios::binary);
		outputFile.write(jpegdatas[i], dataLengths[i]);
	}

	return 0;
}