#include <iostream>
#include <kinect-camera.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>

namespace kinect {
namespace sensor {

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// camera impl
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

libfreenect2::Freenect2 *freenectInstance = nullptr;

libfreenect2::Freenect2* getFreenectInstance()
{
  if (!freenectInstance)
    freenectInstance = new libfreenect2::Freenect2();
  return freenectInstance;
}


struct KinectCameraState
{

  KinectCameraState();
  ~KinectCameraState();
  /** Copy constructor */
  KinectCameraState (const KinectCameraState& other);
  /** Move constructor */
  KinectCameraState (KinectCameraState&& other) noexcept;
  /** Copy assignment operator */
  KinectCameraState& operator= (const KinectCameraState& other);
  /** Move assignment operator */
  KinectCameraState& operator= (KinectCameraState&& other) noexcept;

  // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

  libfreenect2::Freenect2 *freenect; // not managed, just a ref
  libfreenect2::Freenect2Device *cam;
  libfreenect2::SyncMultiFrameListener *listener;
  libfreenect2::Registration *registration;
};

KinectCameraState::KinectCameraState() : freenect(getFreenectInstance())
{
  kinect::sensor::createFreenectDev(*freenect, &cam);
  listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
  cam->setColorFrameListener(listener);
  cam->setIrAndDepthFrameListener(listener);
  cam->start();

  registration = new libfreenect2::Registration(cam->getIrCameraParams(), cam->getColorCameraParams());

  std::cout << "device serial: " << cam->getSerialNumber() << std::endl;
  std::cout << "device firmware: " << cam->getFirmwareVersion() << std::endl;
};

KinectCameraState::~KinectCameraState()
{
  kinect::sensor::stopFreenectDev(cam);
  delete cam;
  cam = nullptr;
  
  delete registration; registration = nullptr;
  delete listener; listener = nullptr;
}

  /** Copy constructor */
KinectCameraState::KinectCameraState (const KinectCameraState& other)
  : freenect(other.freenect)
{
  std::cout << "copying KinectCameraState" << std::endl;
  kinect::sensor::createFreenectDev(*freenect, &cam);
  listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
  cam->setColorFrameListener(listener);
  cam->setIrAndDepthFrameListener(listener);
  cam->start();
  registration = new libfreenect2::Registration(cam->getIrCameraParams(), cam->getColorCameraParams());
}

/** Move constructor */
KinectCameraState::KinectCameraState (KinectCameraState&& other) noexcept
  : freenect(other.freenect), cam(other.cam), listener(other.listener), registration(other.registration)
{ other.cam = nullptr; other.listener = nullptr; other.registration = nullptr; }

/** Copy assignment operator */
KinectCameraState& KinectCameraState::operator= (const KinectCameraState& other)
{
    KinectCameraState tmp(other); // re-use copy-constructor
    *this = std::move(tmp); // re-use move-assignment
    return *this;
}

/** Move assignment operator */
KinectCameraState& KinectCameraState::operator= (KinectCameraState&& other) noexcept
{
  std::swap(freenect, other.freenect);
  std::swap(cam, other.cam);
  std::swap(listener, other.listener);
  std::swap(registration, other.registration);
  return *this;
}





// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// camera interface
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

KinectCamera::KinectCamera()
  : state(new kinect::sensor::KinectCameraState()) {};

void KinectCamera::read(cv::Mat &frame)
{
  size_t width = 512, height = 424;
  libfreenect2::FrameMap frames;;
  libfreenect2::Frame undistorted(width, height, 4);
  libfreenect2::Frame registered(width, height, 4);

  state->listener->waitForNewFrame(frames);
  libfreenect2::Frame *rgbFrame   = frames[libfreenect2::Frame::Color];
  libfreenect2::Frame *irFrame    = frames[libfreenect2::Frame::Ir];
  libfreenect2::Frame *depthFrame = frames[libfreenect2::Frame::Depth];

  // state->registration->apply(rgbFrame, depthFrame, &undistorted, &registered);


  
  state->listener->release(frames);
}

// {
//   frame.create(height, width, CV_8UC3);
//   unsigned char *input = (unsigned char*)(frame.data);

//   float x, y, z, rgb;
//   for (int c = 0; c < 512; c++)
//   {
//     for (int r = 0; r < 424; r++)
//     {
//       state->registration->getPointXYZRGB(&undistorted, &registered, r, c, x, y, z, rgb);
//       if (std::isnan(x)) continue;
      
//       // pcl::PointXYZRGB point;
//       // point.x = x;
//       // point.y = y;
//       // point.z = z;
//       // point.rgb = rgb;
//       uint32_t rgbInt = *reinterpret_cast<int*>(&rgb);
//       uint8_t red = (rgbInt >> 16) & 0x0000ff;
//       // uint8_t red = 200;
//       uint8_t green = (rgbInt >> 8)  & 0x0000ff;
//       uint8_t blue = (rgbInt)       & 0x0000ff;
//       input[height * r + c] = blue;
//       input[height * r + c + 1] = green;
//       input[height * r + c + 2] = red;
//       // input[height * r + c + 3] = 255;
//     }
//   }

//   // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

//   // frame.create(rgb->height, rgb->width, CV_8UC4);
//   // memcpy(frame.data, rgb->data, rgb->height*rgb->width*4*sizeof(uchar));
//   // cv::Mat rgbMat(rgb->height, rgb->width, CV_8UC4, rgb->data);

//   // resize(rgbMat, rgbMat, cv::Size(), 0.5, 0.5);
//   // cv::imshow("rgb", frame);
//   // cv::imshow("rgb", rgbMat);
//   // cv::imshow("ir", cv::Mat(ir->height, ir->width, CV_32FC1, ir->data) / 500.0f);
//   // cv::imshow("depth", cv::Mat(depth->height, depth->width, CV_32FC1, depth->data) / 4500.0f);

//   // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//   // bool stopped = run(rgb, ir, depth, registration, &undistorted, &registered);
//   // ...
// }
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


}
}
