# CMake entry point.
# Note that this CMakeLists.txt in fully independent from the one in the parent directory
# (since that's the goal of this tutorial !)

# For an introduction to CMake, see
# http://www.cmake.org/cmake/help/cmake_tutorial.html (at least steps 1 and 2)

cmake_minimum_required (VERSION 2.6)
project ( GigaRender )

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

# Eigen3
if (WIN32)
	set(EIGEN_INCLUDE_DIR "E:/libs/Eigen/include/eigen3")
elseif ( UNIX )
		set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
endif()
include_directories(${EIGEN_INCLUDE_DIR})

# opengl
find_package(OpenGL REQUIRED)
# Ask the user to give the paths to GLFW, GLEW and GLM.
# On Linux this can be found automatically with pkg-config; 
# see http://stackoverflow.com/questions/6889264/cmake-flags-for-opengl-using-glew-and-glfw, for instance.
if ( WIN32 )
	set(GLFW_INCLUDE_PATH "E:/libs/OpenGL/include/GLFW")
	set(GLFW_LIBRARY "E:/libs/OpenGL/lib/glfw3.lib" )
	set(GLEW_INCLUDE_PATH "E:/libs/OpenGL/include")
	set(GLEW_LIBRARY "E:/libs/OpenGL/lib/GLEW_1130.lib")
	set(GLM_INCLUDE_PATH "E:/libs/OpenGL/include")

	add_definitions(
		#-DTW_STATIC
		#-DTW_NO_LIB_PRAGMA
		#-DTW_NO_DIRECT3D
		-DGLEW_STATIC
		#-D_CRT_SECURE_NO_WARNINGS
	)
	
#	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

#cuda
find_package(CUDA)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
#    -O3 
	-gencode=arch=compute_61,code=\"sm_61,compute_61\"
    )

# OpenMP
if (WIN32)
	if (MSVC)
		#set(CMAKE_CXX_FLAGS_RELEASE "/openmp")
		# set mt for release
	endif(MSVC)
elseif( UNIX )
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

# application lib
include_directories( ${PROJECT_SOURCE_DIR}/applications )
set(USE_LIB false)
if (USE_LIB)
	if (MSVC)
		set(TRACK_LIBRARY_DEBUG "${PROJECT_SOURCE_DIR}/applications/lib/libfast_tracking_sharedd.lib")
		set(TRACK_LIBRARY_RELEASE "${PROJECT_SOURCE_DIR}/applications/lib/libfast_tracking_shared.lib")
	endif(MSVC)
	set(APP_HEADER_FILES
		${PROJECT_SOURCE_DIR}/applications/tracking/tracker.h
	)
	set(APP_SOURCE_FILES 
	)
else()
	set(APP_HEADER_FILES
		${PROJECT_SOURCE_DIR}/applications/tracking/tracker.h
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_datatype.h
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_detect_scale.h
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_detect_trans.h
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_fft.h
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_fhog.h
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_proposal.h
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_table.h
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_utilized.h
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/ValStructVec.h
	)
	set(APP_SOURCE_FILES
		${PROJECT_SOURCE_DIR}/applications/tracking/tracker.cpp
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_detect_scale.cpp
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_detect_trans.cpp
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_fft.cpp
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_fhog.cpp
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_proposal.cpp
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_table.cpp
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_utilized.cpp
		${PROJECT_SOURCE_DIR}/applications/tracking/libTracking/tk_sysapi.cpp
	)
endif(USE_LIB)

set(MY_HEADER_FILES
	# C++ header file
	${PROJECT_SOURCE_DIR}/common/shader.hpp
	${PROJECT_SOURCE_DIR}/common/texture.hpp
	GigaLog.hpp
	VideoIO.h
	imgwarp.h
	GraphCutMask.h
	OGLRender.hpp
	# applications
	${PROJECT_SOURCE_DIR}/applications/AppUtil.h
)

set(MY_SOURCE_FILES 
	# C++ code
	${PROJECT_SOURCE_DIR}/common/shader.cpp
	${PROJECT_SOURCE_DIR}/common/texture.cpp
	main.cpp
	VideoIO.cpp
	imgwarp.cpp
	GraphCutMask.cpp
	# glsl opengl shader
	${PROJECT_SOURCE_DIR}/shader/TransformVertexShader.vertexshader.glsl
	${PROJECT_SOURCE_DIR}/shader/TextureFragmentShader.fragmentshader.glsl
	# cuda code
	cuda/imgwarp.cu
	# applications
	${PROJECT_SOURCE_DIR}/applications/AppUtil.cpp
	${PROJECT_SOURCE_DIR}/applications/AppUtil.cu
)

source_group("common" FILES 
	${PROJECT_SOURCE_DIR}/common/shader.hpp
	${PROJECT_SOURCE_DIR}/common/texture.hpp
)
source_group("shader"  FILES 
	${PROJECT_SOURCE_DIR}/shader/TransformVertexShader.vertexshader.glsl
	${PROJECT_SOURCE_DIR}/shader/TextureFragmentShader.fragmentshader.glsl
	${PROJECT_SOURCE_DIR}/shader/SolidColor.fragmentshader.glsl
)
source_group("cuda"  FILES 
	cuda/imgwarp.cu
)
source_group("AppUtil"  FILES 
	${PROJECT_SOURCE_DIR}/applications/AppUtil.cpp
	${PROJECT_SOURCE_DIR}/applications/AppUtil.cu
	${PROJECT_SOURCE_DIR}/applications/AppUtil.h
)
source_group("Apps"  FILES 
	${APP_HEADER_FILES}
	${APP_SOURCE_FILES}
)

include_directories(
	${GLFW_INCLUDE_PATH}
	${GLEW_INCLUDE_PATH}
	${GLM_INCLUDE_PATH}
	${PROJECT_SOURCE_DIR}	
	${PROJECT_SOURCE_DIR}/common	
	${PROJECT_SOURCE_DIR}/cuda	
)
cuda_add_executable(GigaRender
	${MY_HEADER_FILES}
	${MY_SOURCE_FILES}
	${APP_HEADER_FILES}
	${APP_SOURCE_FILES}
)


if (USE_LIB)
	target_link_libraries(GigaRender
		${OPENGL_LIBRARY} # filled by "find_package(OpenGL REQUIRED)"
		${GLFW_LIBRARY}
		${GLEW_LIBRARY}
		${OpenCV_LIBS}
		debug ${TRACK_LIBRARY_DEBUG}
		optimized ${TRACK_LIBRARY_RELEASE}
	)
else()
	target_link_libraries(GigaRender
		${OPENGL_LIBRARY} # filled by "find_package(OpenGL REQUIRED)"
		${GLFW_LIBRARY}
		${GLEW_LIBRARY}
		${OpenCV_LIBS}
	)
endif(USE_LIB)