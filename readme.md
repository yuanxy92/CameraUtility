# PointGrey Camera Array Utility

## Updated Oct 14, 2017
add NPPJpegCoder class. Now this code can capture images and compress them in jpeg format in real time. 
- GTX 1070 costs 50ms to compress 8 4000x3000 images
- GTX 1080Ti + i7 6800 could capture 8 images and compress them in about 40ms

## Libraries
- Spinnaker SDK (downloaded from FLIR/Pointgrey website)
- OpenCV >= 3.0
- Windows Visual Studio 2015 (Linux or other Visual Studio Version should be OK, but untested, C++11 feature is needed for multi-threads feature ), do not use Visual Studio 2017 now!
- CUDA 8.0 (I use cuda code to do naive demosaicing, you can use a CPU function to replace it)

## Notice
I use FL3-U3-120S3C camera with 4000x3000 resolution. Please modify the code here:

CameraArray.cpp line_48, the image resolution parameters
```
for (size_t i = 0; i < frameNum; i++) {
	bufferImgs[i].resize(camutil.getCameraNum());
	for (size_t j = 0; j < camutil.getCameraNum(); j++) {
		bufferImgs[i][j].create(3000, 4000, CV_8U);
	}
}
```

There may be other bugs. #^_^#

Xiaoyun YUAN

