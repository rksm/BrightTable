#include <iostream>
#include <fstream>

#include <kinect-camera.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>

size_t IR_IMAGE_WIDTH = 512, IR_IMAGE_HEIGHT = 424;

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
  size_t width = 1920, height = 1080;
  libfreenect2::FrameMap frames;;
  libfreenect2::Frame undistorted(IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT, 4);
  libfreenect2::Frame registered(IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT, 4);

  state->listener->waitForNewFrame(frames);
  libfreenect2::Frame *rgbFrame   = frames[libfreenect2::Frame::Color];
  libfreenect2::Frame *irFrame    = frames[libfreenect2::Frame::Ir];
  libfreenect2::Frame *depthFrame = frames[libfreenect2::Frame::Depth];

  state->registration->apply(rgbFrame, depthFrame, &undistorted, &registered);
  
  frame.create(rgbFrame->height, rgbFrame->width, CV_8UC4);
  memcpy(frame.data, rgbFrame->data, rgbFrame->height*rgbFrame->width*4*sizeof(uchar));

  state->listener->release(frames);
}

void KinectCamera::readWithDepth(cv::Mat &frame, cv::Mat &depth)
{
  size_t width = 1920, height = 1080;
  libfreenect2::FrameMap frames;;
  libfreenect2::Frame undistorted(IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT, 4);
  libfreenect2::Frame registered(IR_IMAGE_WIDTH, IR_IMAGE_HEIGHT, 4);
  libfreenect2::Frame bigDepthFrame(1920,1080+2,4);

  state->listener->waitForNewFrame(frames);
  libfreenect2::Frame *rgbFrame   = frames[libfreenect2::Frame::Color];
  libfreenect2::Frame *irFrame    = frames[libfreenect2::Frame::Ir];
  libfreenect2::Frame *depthFrame = frames[libfreenect2::Frame::Depth];

  state->registration->apply(rgbFrame, depthFrame, &undistorted, &registered, true, &bigDepthFrame);
  
  frame.create(height, width, CV_8UC4);
  memcpy(frame.data, rgbFrame->data, height*width*4*sizeof(uchar));

  depth.create(height, width, CV_32FC1);
  memcpy(depth.data, bigDepthFrame.data, height*width*1*sizeof(float));

  // imshow("depth", depth / 1000.0f);
  // imwrite("depth.exr", depth / 1000.0f);

  if (false)
  {
    float max = 0, min = 9999;
    std::stringstream ss;
    for (int i = 0; i < height; i++)
    {
      for (int j = 0; j < width; j++)
      {
        ss << depth.at<float>(i,j) << ",";
        // float val = depth.at<float>(i,j);
        // if (max < val) max = val;
        // if (min > val && val != 0) min = val;
        // if (val > 255) val = 255;
      
        // depth.at<cv::Vec4b>(i,j)[0] = val;
        // depth.at<cv::Vec4b>(i,j)[1] = val;
        // depth.at<cv::Vec4b>(i,j)[2] = val;
      }
      ss << "\n";
    }
    std::cout << min << "/" << max << std::endl;
  
    std::ofstream myfile;
    myfile.open("depth.txt");
    myfile << ss.str();
  }

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
