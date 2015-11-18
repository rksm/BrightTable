#include <iostream>
#include <mutex>

#include <opencv2/opencv.hpp>

#include <camera.hpp>
#include <kinect-camera.hpp>

namespace vision {
namespace cam {

CvCamera::CvCamera(int devNo) : cam(new cv::VideoCapture(devNo)) {};
void CvCamera::read(cv::Mat &frame) { cam->read(frame); };
void CvCamera::readWithDepth(cv::Mat &frame, cv::Mat &depth) { return read(frame); };
bool CvCamera::isOpen() { return cam->isOpened(); };

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

std::mutex camsMutex;
Cameras cams;

CameraPtr cameraFindOrInsert(std::string key, std::function <CameraPtr()> createFunc)
{
  camsMutex.lock();
  CameraPtr cam;
  auto camIt = cams.find(key);
  if (camIt != cams.end()) {
    cam = camIt->second;
  } else {
    cam = createFunc();
    cams.insert({key, cam});
  }
  camsMutex.unlock();
  return cam;
}

CameraPtr getCamera(int devNo)
{
  return cameraFindOrInsert(
    std::to_string(devNo),
    [&]() -> CameraPtr { return std::make_shared<CvCamera>(CvCamera(devNo)); });
}

CameraPtr getCamera(std::string key)
{
  std::cout << "kinect action? " << key << std::endl;
  if (key == "kinect") {
    return cameraFindOrInsert(
      key,
      [&]() -> CameraPtr {
        return std::make_shared<kinect::sensor::KinectCamera>(); });
  }
  throw std::invalid_argument(key);
}

}
}
