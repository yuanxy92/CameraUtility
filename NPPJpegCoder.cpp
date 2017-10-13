/*
@brief c++ source class for NPP jpeg coder
@author Shane Yuan
@date Oct 11, 2017
*/

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <string>
#include <fstream>
#include <iostream>

#include "NPPJpegCoderKernel.h"

#include "Exceptions.h"
#include "NPPJpegCoder.h"

#include "helper_string.h"
#include "helper_cuda.h"

using namespace std;

//#define MEASURE_KERNEL_TIME

namespace npp {

	template<class T>
	T readBigEndian(const unsigned char *pData) {
		if (sizeof(T) > 1)
		{
			unsigned char p[sizeof(T)];
			reverse_copy(pData, pData + sizeof(T), p);
			return *reinterpret_cast<T *>(p);
		}
		else
		{
			return *pData;
		}
	}

	template<class T>
	void writeBigEndian(unsigned char *pData, T value)
	{
		unsigned char *pValue = reinterpret_cast<unsigned char *>(&value);
		reverse_copy(pValue, pValue + sizeof(T), pData);
	}

	int DivUp(int x, int d) {
		return (x + d - 1) / d;
	}

	template<typename T>
	T readAndAdvance(const unsigned char *&pData) {
		T nElement = readBigEndian<T>(pData);
		pData += sizeof(T);
		return nElement;
	}

	template<typename T>
	void writeAndAdvance(unsigned char *&pData, T nElement) {
		writeBigEndian<T>(pData, nElement);
		pData += sizeof(T);
	}

	int nextMarker(const unsigned char *pData, int &nPos, int nLength) {
		unsigned char c = pData[nPos++];

		do
		{
			while (c != 0xffu && nPos < nLength)
			{
				c = pData[nPos++];
			}

			if (nPos >= nLength)
				return -1;

			c = pData[nPos++];
		} while (c == 0 || c == 0x0ffu);

		return c;
	}

	void writeMarker(unsigned char nMarker, unsigned char *&pData) {
		*pData++ = 0x0ff;
		*pData++ = nMarker;
	}

	void writeJFIFTag(unsigned char *&pData) {
		const char JFIF_TAG[] =
		{
			0x4a, 0x46, 0x49, 0x46, 0x00,
			0x01, 0x02,
			0x00,
			0x00, 0x01, 0x00, 0x01,
			0x00, 0x00
		};

		writeMarker(0x0e0, pData);
		writeAndAdvance<unsigned short>(pData, sizeof(JFIF_TAG) + sizeof(unsigned short));
		memcpy(pData, JFIF_TAG, sizeof(JFIF_TAG));
		pData += sizeof(JFIF_TAG);
	}

	void loadJpeg(const char *input_file, unsigned char *&pJpegData, int &nInputLength) {
		// Load file into CPU memory
		ifstream stream(input_file, ifstream::binary);

		if (!stream.good())
		{
			return;
		}

		stream.seekg(0, ios::end);
		nInputLength = (int)stream.tellg();
		stream.seekg(0, ios::beg);

		pJpegData = new unsigned char[nInputLength];
		stream.read(reinterpret_cast<char *>(pJpegData), nInputLength);
	}

	void readFrameHeader(const unsigned char *pData, FrameHeader &header) {
		readAndAdvance<unsigned short>(pData);
		header.nSamplePrecision = readAndAdvance<unsigned char>(pData);
		header.nHeight = readAndAdvance<unsigned short>(pData);
		header.nWidth = readAndAdvance<unsigned short>(pData);
		header.nComponents = readAndAdvance<unsigned char>(pData);

		for (int c = 0; c < header.nComponents; ++c)
		{
			header.aComponentIdentifier[c] = readAndAdvance<unsigned char>(pData);
			header.aSamplingFactors[c] = readAndAdvance<unsigned char>(pData);
			header.aQuantizationTableSelector[c] = readAndAdvance<unsigned char>(pData);
		}

	}

	void writeFrameHeader(const FrameHeader &header, unsigned char *&pData) {
		unsigned char aTemp[128];
		unsigned char *pTemp = aTemp;

		writeAndAdvance<unsigned char>(pTemp, header.nSamplePrecision);
		writeAndAdvance<unsigned short>(pTemp, header.nHeight);
		writeAndAdvance<unsigned short>(pTemp, header.nWidth);
		writeAndAdvance<unsigned char>(pTemp, header.nComponents);

		for (int c = 0; c < header.nComponents; ++c)
		{
			writeAndAdvance<unsigned char>(pTemp, header.aComponentIdentifier[c]);
			writeAndAdvance<unsigned char>(pTemp, header.aSamplingFactors[c]);
			writeAndAdvance<unsigned char>(pTemp, header.aQuantizationTableSelector[c]);
		}

		unsigned short nLength = (unsigned short)(pTemp - aTemp);

		writeMarker(0x0C0, pData);
		writeAndAdvance<unsigned short>(pData, nLength + 2);
		memcpy(pData, aTemp, nLength);
		pData += nLength;
	}


	void readScanHeader(const unsigned char *pData, ScanHeader &header) {
		readAndAdvance<unsigned short>(pData);

		header.nComponents = readAndAdvance<unsigned char>(pData);

		for (int c = 0; c < header.nComponents; ++c)
		{
			header.aComponentSelector[c] = readAndAdvance<unsigned char>(pData);
			header.aHuffmanTablesSelector[c] = readAndAdvance<unsigned char>(pData);
		}

		header.nSs = readAndAdvance<unsigned char>(pData);
		header.nSe = readAndAdvance<unsigned char>(pData);
		header.nA = readAndAdvance<unsigned char>(pData);
	}


	void writeScanHeader(const ScanHeader &header, unsigned char *&pData) {
		unsigned char aTemp[128];
		unsigned char *pTemp = aTemp;

		writeAndAdvance<unsigned char>(pTemp, header.nComponents);

		for (int c = 0; c < header.nComponents; ++c)
		{
			writeAndAdvance<unsigned char>(pTemp, header.aComponentSelector[c]);
			writeAndAdvance<unsigned char>(pTemp, header.aHuffmanTablesSelector[c]);
		}

		writeAndAdvance<unsigned char>(pTemp, header.nSs);
		writeAndAdvance<unsigned char>(pTemp, header.nSe);
		writeAndAdvance<unsigned char>(pTemp, header.nA);

		unsigned short nLength = (unsigned short)(pTemp - aTemp);

		writeMarker(0x0DA, pData);
		writeAndAdvance<unsigned short>(pData, nLength + 2);
		memcpy(pData, aTemp, nLength);
		pData += nLength;
	}


	void readQuantizationTables(const unsigned char *pData, QuantizationTable *pTables) {
		unsigned short nLength = readAndAdvance<unsigned short>(pData) - 2;

		while (nLength > 0)
		{
			unsigned char nPrecisionAndIdentifier = readAndAdvance<unsigned char>(pData);
			int nIdentifier = nPrecisionAndIdentifier & 0x0f;

			pTables[nIdentifier].nPrecisionAndIdentifier = nPrecisionAndIdentifier;
			memcpy(pTables[nIdentifier].aTable, pData, 64);
			pData += 64;

			nLength -= 65;
		}
	}

	void writeQuantizationTable(const QuantizationTable &table, unsigned char *&pData) {
		writeMarker(0x0DB, pData);
		writeAndAdvance<unsigned short>(pData, sizeof(QuantizationTable) + 2);
		memcpy(pData, &table, sizeof(QuantizationTable));
		pData += sizeof(QuantizationTable);
	}

	void readHuffmanTables(const unsigned char *pData, HuffmanTable *pTables) {
		unsigned short nLength = readAndAdvance<unsigned short>(pData) - 2;

		while (nLength > 0)
		{
			unsigned char nClassAndIdentifier = readAndAdvance<unsigned char>(pData);
			int nClass = nClassAndIdentifier >> 4; // AC or DC
			int nIdentifier = nClassAndIdentifier & 0x0f;
			int nIdx = nClass * 2 + nIdentifier;
			pTables[nIdx].nClassAndIdentifier = nClassAndIdentifier;

			// Number of Codes for Bit Lengths [1..16]
			int nCodeCount = 0;

			for (int i = 0; i < 16; ++i)
			{
				pTables[nIdx].aCodes[i] = readAndAdvance<unsigned char>(pData);
				nCodeCount += pTables[nIdx].aCodes[i];
			}

			memcpy(pTables[nIdx].aTable, pData, nCodeCount);
			pData += nCodeCount;

			nLength -= 17 + nCodeCount;
		}
	}

	void writeHuffmanTable(const HuffmanTable &table, unsigned char *&pData) {
		writeMarker(0x0C4, pData);

		// Number of Codes for Bit Lengths [1..16]
		int nCodeCount = 0;

		for (int i = 0; i < 16; ++i)
		{
			nCodeCount += table.aCodes[i];
		}

		writeAndAdvance<unsigned short>(pData, 17 + nCodeCount + 2);
		memcpy(pData, &table, 17 + nCodeCount);
		pData += 17 + nCodeCount;
	}


	void readRestartInterval(const unsigned char *pData, int &nRestartInterval) {
		readAndAdvance<unsigned short>(pData);
		nRestartInterval = readAndAdvance<unsigned short>(pData);
	}

	bool printfNPPinfo(int cudaVerMajor, int cudaVerMinor) {
		const NppLibraryVersion *libVer = nppGetLibVersion();

		printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

		int driverVersion, runtimeVersion;
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);

		printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
		printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

		bool bVal = checkCudaCapabilities(cudaVerMajor, cudaVerMinor);
		return bVal;
	}

	/**
	@brief constructor
	*/
	NPPJpegCoder::NPPJpegCoder() {}
	NPPJpegCoder::~NPPJpegCoder() {}

	/**
	@brief init jpeg encoder
	@param int width: input image width
	@param int height: input image height
	@param int quality: jpeg encoding quality
	@return
	*/
	int NPPJpegCoder::init(int width, int height, int quality) {
		if (printfNPPinfo(2, 0) == false) {
			cerr << "jpegNPP requires a GPU with Compute Capability 2.0 or higher" << endl;
			exit(-1);
		}
		// calculate quantization table from quality
		float s;
		if (quality < 50) 
			s = 5000.0f / quality;
		else s = 200.0f - 2 * quality;
		for (size_t i = 0; i < 64; i++) {
			// luminance
			float luminVal = (float)quantiztionTableLuminance[i];
			luminVal = floor((s * luminVal + 50.0f) / 100.0f);
			if (luminVal < 1)
				luminVal = 1;
			else if (luminVal > 255)
				luminVal = 255;
			quantiztionTableLuminance[i] = (unsigned char)luminVal;
			// chroma
			float chromaVal = (float)quantiztionTableChroma[i];
			chromaVal = floor((s * chromaVal + 50.0f) / 100.0f);
			if (chromaVal < 1)
				chromaVal = 1;
			else if (chromaVal > 255)
				chromaVal = 255;
			quantiztionTableChroma[i] = (unsigned char)chromaVal;
		}

		// set width and height
		this->width = width;
		this->height = height;
		// calculate size
		aSrcSize[0].width = width;
		aSrcSize[0].height = height;
		aSrcSize[1].width = width;
		aSrcSize[1].height = height;
		aSrcSize[2].width = width;
		aSrcSize[2].height = height;
		aDstSize[0].width = width;
		aDstSize[0].height = height;
		aDstSize[1].width = width / 2;
		aDstSize[1].height = height / 2;
		aDstSize[2].width = width / 2;
		aDstSize[2].height = height / 2;
		// init output frame header
		memset(&oFrameHeader, 0, sizeof(FrameHeader));
		oFrameHeader.nWidth = width;
		oFrameHeader.nHeight = height;
		oFrameHeader.nSamplePrecision = 8;
		oFrameHeader.nComponents = 3;
		oFrameHeader.aComponentIdentifier[0] = 1;
		oFrameHeader.aComponentIdentifier[1] = 2;
		oFrameHeader.aComponentIdentifier[2] = 3;
		oFrameHeader.aSamplingFactors[0] = 34;
		oFrameHeader.aSamplingFactors[1] = 17;
		oFrameHeader.aSamplingFactors[2] = 17;
		oFrameHeader.aQuantizationTableSelector[0] = 0;
		oFrameHeader.aQuantizationTableSelector[1] = 1;
		oFrameHeader.aQuantizationTableSelector[2] = 1;
		// init quantization table
		memset(aQuantizationTables, 0, 4 * sizeof(QuantizationTable));
		memset(aHuffmanTables, 0, 4 * sizeof(HuffmanTable));
		aQuantizationTables[0].nPrecisionAndIdentifier = 0;
		memcpy(aQuantizationTables[0].aTable, quantiztionTableLuminance, 64 * sizeof(unsigned char));
		aQuantizationTables[1].nPrecisionAndIdentifier = 1;
		memcpy(aQuantizationTables[1].aTable, quantiztionTableChroma, 64 * sizeof(unsigned char));
		cudaMalloc(&pdQuantizationTables, 64 * 4);
		// Copy DCT coefficients and Quantization Tables from host to device
		for (int i = 0; i < 2; ++i) {
			NPP_CHECK_CUDA(cudaMemcpy(pdQuantizationTables + i * 64,
				aQuantizationTables[i].aTable, 64, cudaMemcpyHostToDevice));
		}
		// init huffman table
		memcpy(aHuffmanTables[0].aCodes, huffmanCodeLuminanceDC, 16 * sizeof(unsigned char));
		memcpy(aHuffmanTables[0].aTable, huffmanTableLuminanceDC, 256 * sizeof(unsigned char));
		aHuffmanTables[0].nClassAndIdentifier = 0;
		memcpy(aHuffmanTables[1].aCodes, huffmanCodeChromaDC, 16 * sizeof(unsigned char));
		memcpy(aHuffmanTables[1].aTable, huffmanTableChromaDC, 256 * sizeof(unsigned char));
		aHuffmanTables[1].nClassAndIdentifier = 1;
		memcpy(aHuffmanTables[2].aCodes, huffmanCodeLuminanceAC, 16 * sizeof(unsigned char));
		memcpy(aHuffmanTables[2].aTable, huffmanTableLuminanceAC, 256 * sizeof(unsigned char));
		aHuffmanTables[2].nClassAndIdentifier = 16;
		memcpy(aHuffmanTables[3].aCodes, huffmanCodeChromaAC, 16 * sizeof(unsigned char));
		memcpy(aHuffmanTables[3].aTable, huffmanTableChromaAC, 256 * sizeof(unsigned char));
		aHuffmanTables[3].nClassAndIdentifier = 17;
		pHuffmanDCTables = aHuffmanTables;
		pHuffmanACTables = &aHuffmanTables[2];
		// init scanner header
		oScanHeader.nA = 0;
		oScanHeader.nComponents = 3;
		oScanHeader.nSe = 63;
		oScanHeader.nSs = 0;
		oScanHeader.aComponentSelector[0] = 1;
		oScanHeader.aComponentSelector[1] = 2;
		oScanHeader.aComponentSelector[2] = 3;
		oScanHeader.aHuffmanTablesSelector[0] = 0;
		oScanHeader.aHuffmanTablesSelector[1] = 17;
		oScanHeader.aHuffmanTablesSelector[2] = 17;

		// init nppiEncodeHuffmanSpecInitAlloc_JPEG
		for (int i = 0; i < 3; ++i) {
			nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, nppiDCTable, &apHuffmanDCTable[i]);
			nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f)].aCodes, nppiACTable, &apHuffmanACTable[i]);
		}
		
		int nMCUBlocksH = 0;
		int nMCUBlocksV = 0;

		// Compute channel sizes as stored in the JPEG (8x8 blocks & MCU block layout)
		for (int i = 0; i < oFrameHeader.nComponents; ++i) {
			nMCUBlocksV = max(nMCUBlocksV, oFrameHeader.aSamplingFactors[i] & 0x0f);
			nMCUBlocksH = max(nMCUBlocksH, oFrameHeader.aSamplingFactors[i] >> 4);
		}

		for (int i = 0; i < oFrameHeader.nComponents; ++i) {
			NppiSize oBlocks;
			NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i] >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f };
			oBlocks.width = (int)ceil((oFrameHeader.nWidth + 7) / 8 *
				static_cast<float>(oBlocksPerMCU.width) / nMCUBlocksH);
			oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;
			oBlocks.height = (int)ceil((oFrameHeader.nHeight + 7) / 8 *
				static_cast<float>(oBlocksPerMCU.height) / nMCUBlocksV);
			oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;
			aDstSize[i].width = oBlocks.width * 8;
			aDstSize[i].height = oBlocks.height * 8;
			// Allocate Memory
			size_t nPitch;
			NPP_CHECK_CUDA(cudaMallocPitch(&apdDCT[i], &nPitch, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height));
			aDCTStep[i] = static_cast<Npp32s>(nPitch);

			NPP_CHECK_CUDA(cudaMallocPitch(&apDstImage[i], &nPitch, aDstSize[i].width, aDstSize[i].height));
			aDstImageStep[i] = static_cast<Npp32s>(nPitch);

			NPP_CHECK_CUDA(cudaHostAlloc(&aphDCT[i], aDCTStep[i] * oBlocks.height, cudaHostAllocDefault));

			pitch[i] = nPitch;
		}

		
		// Huffman Encoding
		NPP_CHECK_CUDA(cudaMalloc(&pdScan, sizeof(unsigned char) * 1024 * 1024 * 10));
		NPP_CHECK_NPP(nppiEncodeHuffmanGetSize(aSrcSize[0], 3, &nTempSize));
		NPP_CHECK_CUDA(cudaMalloc(&pJpegEncoderTemp, nTempSize));

		return 0;
	}

	/**
	@brief release jpeg encode
	@return int
	*/
	int NPPJpegCoder::release() {
		// release memory
		for (int i = 0; i < 3; ++i) {
			nppiEncodeHuffmanSpecFree_JPEG(apHuffmanDCTable[i]);
			nppiEncodeHuffmanSpecFree_JPEG(apHuffmanACTable[i]);
			cudaFree(apdDCT[i]);
			cudaFreeHost(aphDCT[i]);
			cudaFree(apDstImage[i]);
		}
		cudaFree(pJpegEncoderTemp);
		cudaFree(pdQuantizationTables);
		cudaFree(pdScan);
		return 0;
	}

	/**
	@brief encode raw image data to jpeg
	@param cv::Mat bayerRGImg: input bayer image
	@param char* jpegdata: output jpeg data
	@param int* datalength: output data length
	@return
	*/
	int NPPJpegCoder::encode(cv::Mat bayerRGImg, unsigned char* jpegdata, int* datalength) {
		NppiDCTState *pDCTState;

#ifdef MEASURE_KERNEL_TIME
		cudaEvent_t start, stop;
		float elapsedTime;
		cudaEventCreate(&start);
		cudaEventRecord(start, 0);
#endif

		Npp8u* bayer_img_d;
		Npp8u* rgb_img_d;
		int step_rgb;
		cudaMalloc(&bayer_img_d, sizeof(unsigned char) * width * height);
		cudaMemcpy(bayer_img_d, bayerRGImg.data, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);
		rgb_img_d = nppiMalloc_8u_C3(width, height, &step_rgb);
		// debayer
		NppiSize osize;
		osize.width = bayerRGImg.cols;
		osize.height = bayerRGImg.rows;
		NppiRect orect;
		orect.x = 0;
		orect.y = 0;
		orect.width = bayerRGImg.cols;
		orect.height = bayerRGImg.rows;
		// bayer to rgb
		NPP_CHECK_NPP(nppiCFAToRGB_8u_C1C3R(bayer_img_d, bayerRGImg.cols, osize,
			orect, rgb_img_d, step_rgb, NPPI_BAYER_RGGB, NPPI_INTER_UNDEFINED));

		// rgb to yuv420
		NPP_CHECK_NPP(nppiRGBToYUV420_8u_C3P3R(rgb_img_d, step_rgb, apDstImage, aDstImageStep,
			osize));

		NPP_CHECK_NPP(nppiDCTInitAlloc(&pDCTState));
		// Forward DCT
		for (int i = 0; i < 3; ++i) {
			NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(apDstImage[i], aDstImageStep[i],
				apdDCT[i], aDCTStep[i],
				pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[i] * 64,
				aDstSize[i],
				pDCTState));
		}

		NPP_CHECK_NPP(nppiEncodeHuffmanScan_JPEG_8u16s_P3R(apdDCT, aDCTStep,
			0, oScanHeader.nSs, oScanHeader.nSe, oScanHeader.nA >> 4, oScanHeader.nA & 0x0f,
			pdScan, &nScanLength,
			apHuffmanDCTable,
			apHuffmanACTable,
			aDstSize,
			pJpegEncoderTemp));
		
		// Write JPEG
		unsigned char *pDstOutput = jpegdata;

		writeMarker(0x0D8, pDstOutput);
		writeJFIFTag(pDstOutput);
		writeQuantizationTable(aQuantizationTables[0], pDstOutput);
		writeQuantizationTable(aQuantizationTables[1], pDstOutput);
		writeFrameHeader(oFrameHeader, pDstOutput);
		writeHuffmanTable(pHuffmanDCTables[0], pDstOutput);
		writeHuffmanTable(pHuffmanACTables[0], pDstOutput);
		writeHuffmanTable(pHuffmanDCTables[1], pDstOutput);
		writeHuffmanTable(pHuffmanACTables[1], pDstOutput);
		writeScanHeader(oScanHeader, pDstOutput);

		NPP_CHECK_CUDA(cudaMemcpy(pDstOutput, pdScan, nScanLength, cudaMemcpyDeviceToHost));

		pDstOutput += nScanLength;
		writeMarker(0x0D9, pDstOutput);

#ifdef MEASURE_KERNEL_TIME
		cudaEventCreate(&stop);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("JPEG encode: (file:%s, line:%d) elapsed time : %f ms\n", __FILE__, __LINE__, elapsedTime);
#endif
		
		// calculate compressed jpeg data length
		*datalength = static_cast<size_t>(pDstOutput - jpegdata);
		// release gpu memory
		nppiDCTFree(pDCTState);
		cudaFree(bayer_img_d);
		nppiFree(rgb_img_d);

		return 0;
	}
};


