#include <iostream>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>

#include "kinect-sensor.hpp"

using libfreenect2::Freenect2;
using libfreenect2::Freenect2Device;
using libfreenect2::Registration;
using libfreenect2::Frame;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

namespace kinect {
namespace sensor {

void
createFreenectDev(
  Freenect2 &freenect2,
  Freenect2Device **dev)
{
  if (freenect2.enumerateDevices() == 0) return;
  std::string serial = freenect2.getDefaultDeviceSerialNumber();
  libfreenect2::OpenCLPacketPipeline *pipeline = new libfreenect2::OpenCLPacketPipeline();
  *dev = freenect2.openDevice(serial, pipeline);
}

void
stopFreenectDev(Freenect2Device *dev)
{
  dev->stop();
  dev->close();
}

void
freenectCaptureLoop(
  Freenect2Device *dev,
  std::function<bool (Frame*, Frame*, Frame*, Registration*, Frame*, Frame*)> run)
{
  libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
  libfreenect2::FrameMap frames;
  libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4);

  dev->setColorFrameListener(&listener);
  dev->setIrAndDepthFrameListener(&listener);
  dev->start();

  std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
  std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

  auto registration = new Registration(dev->getIrCameraParams(), dev->getColorCameraParams());

  while(1)
  {
    listener.waitForNewFrame(frames);
    libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    registration->apply(rgb, depth, &undistorted, &registered);

    bool stopped = run(rgb, ir, depth, registration, &undistorted, &registered);
    
    listener.release(frames);

    if (stopped) break;
  }
  
  stopFreenectDev(dev);
  delete registration;
}

}
}
