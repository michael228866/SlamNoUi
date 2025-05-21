cmake_minimum_required(VERSION 3.10)
project(slam_from_live)

# C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "F:/opencv/build/x64/vc16/lib")

# 找 OpenCV
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

# Include paths
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/aruco/aruco)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/eigen3)
include_directories(${CMAKE_SOURCE_DIR}/tests)

# 目標執行檔
add_executable(slam_from_live
    slam_from_live.cpp

    # aruco 源碼
    aruco/aruco/markerdetector.cpp
    aruco/aruco/markerdetector_impl.cpp
    aruco/aruco/marker.cpp
    aruco/aruco/markerlabeler.cpp
    aruco/aruco/dictionary.cpp
    aruco/aruco/dictionary_based.cpp
    aruco/aruco/markermap.cpp
    aruco/aruco/cameraparameters.cpp
    aruco/aruco/debug.cpp
    aruco/aruco/ippe.cpp

    tests/utils3d.cpp
)

# include dirs for target
target_include_directories(slam_from_live PRIVATE
    ${CMAKE_SOURCE_DIR}/aruco/aruco
    ${CMAKE_SOURCE_DIR}/aruco/utils
    ${CMAKE_SOURCE_DIR}/tests
    ${CURL_INCLUDE_DIRS}  # ✅ 加入 curl 的 headers

)

target_link_libraries(slam_from_live PRIVATE ${OpenCV_LIBS} ${CURL_LIBRARIES})
