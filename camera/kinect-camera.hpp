#ifndef HAND_DETECTOR_SERVER_KINECT_CAMERA_STATE_H_
#define HAND_DETECTOR_SERVER_KINECT_CAMERA_STATE_H_

#include <camera.hpp>
#include <kinect-sensor.hpp>

namespace kinect {
namespace sensor {

struct KinectCameraState;

class KinectCamera : public vision::cam::Camera
{
  public:
    KinectCamera();
    void read(cv::Mat &frame);
  private:
    std::shared_ptr<KinectCameraState> state;
};

}
}

#endif  // HAND_DETECTOR_SERVER_KINECT_CAMERA_STATE_H_