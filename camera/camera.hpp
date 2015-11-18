#ifndef HAND_DETECTOR_SERVER_CAMERA_H_
#define HAND_DETECTOR_SERVER_CAMERA_H_

#include <opencv2/opencv.hpp>

namespace vision {
namespace cam {

class Camera
{
  public:
    virtual void read(cv::Mat &frame) {};
    virtual void readWithDepth(cv::Mat &frame, cv::Mat &depth) {};
    virtual bool isOpen() { return true; };
};

class CvCamera : public Camera
{
  public:
    CvCamera(int devNo);
    void read(cv::Mat&);
    void readWithDepth(cv::Mat &frame, cv::Mat &depth);
    bool isOpen();
  private:
    std::unique_ptr<cv::VideoCapture> cam;
};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

typedef std::shared_ptr<Camera> CameraPtr;
typedef std::map<std::string, CameraPtr> Cameras;

CameraPtr getCamera(int devNo);
CameraPtr getCamera(std::string key);

}
}

#endif  // HAND_DETECTOR_SERVER_CAMERA_H_
