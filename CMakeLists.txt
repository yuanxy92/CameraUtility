# CMake entry point.
# Note that this CMakeLists.txt in fully independent from the one in the parent directory
# (since that's the goal of this tutorial !)

# For an introduction to CMake, see
# http://www.cmake.org/cmake/help/cmake_tutorial.html (at least steps 1 and 2)

cmake_minimum_required (VERSION 2.6)
set(name "CameraUtil")
project ( ${name} )

#opencv 
if ( WIN32 )
	set(OpenCV_DIR "E:/libs/OpenCV/x64/vc14/lib")
	set(OpenCV_CONFIG_PATH "E:/libs/OpenCV")
	set(WIN_HEADER_PATH "${PROJECT_SOURCE_DIR}/winheadfile")
    include_directories(${WIN_HEADER_PATH})
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# pointgrey camera sdk
set(USE_PTGREY_CAMERA on)
if (USE_PTGREY_CAMERA)
	set(PTGREY_SDK_INCLUDE_DIR "C:/Program Files/Point Grey Research/Spinnaker/include")
	set(PTGREY_SDK_LIBRARY_DEBUG "C:/Program Files/Point Grey Research/Spinnaker/lib64/vs2015/Spinnakerd_v140.lib")
	set(PTGREY_SDK_LIBRARY_RELEASE "C:/Program Files/Point Grey Research/Spinnaker/lib64/vs2015/Spinnaker_v140.lib")
endif()
include_directories(${PTGREY_SDK_INCLUDE_DIR})

#cuda
find_package(CUDA)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
#    -O3 
	-gencode=arch=compute_61,code=\"sm_61,compute_61\"
    )

set(MY_HEADER_FILES
	# C++ header file
	cuda/CameraUtilKernel.h
	CameraUtil.h
	CameraArray.h
)

set(MY_SOURCE_FILES 
	# C++ source file
	CameraUtil.cpp
	CameraArray.cpp
	main.cpp
	# cuda source file
	cuda/CameraUtilKernel.cu
)

include_directories(
	${PROJECT_SOURCE_DIR}	
	${PROJECT_SOURCE_DIR}/cuda	
)

cuda_add_executable(${name}
	${MY_HEADER_FILES}
	${MY_SOURCE_FILES}
)

if (USE_LIB)
	target_link_libraries(${name}
		${OpenCV_LIBS}
		debug ${PTGREY_SDK_LIBRARY_DEBUG}
		optimized ${PTGREY_SDK_LIBRARY_RELEASE}
	)
else()
	target_link_libraries(${name}
		${OpenCV_LIBS}
		debug ${PTGREY_SDK_LIBRARY_DEBUG}
		optimized ${PTGREY_SDK_LIBRARY_RELEASE}
	)
endif(USE_LIB)