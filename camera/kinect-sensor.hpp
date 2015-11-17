#ifndef KINECT_TEST_FREENECT_H_INCLUDED
#define KINECT_TEST_FREENECT_H_INCLUDED

#include <functional>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/registration.h>

namespace kinect {
namespace sensor {

void createFreenectDev(libfreenect2::Freenect2&, libfreenect2::Freenect2Device**);

void stopFreenectDev(libfreenect2::Freenect2Device *dev);

void freenectCaptureLoop(
  libfreenect2::Freenect2Device*,
  std::function<bool (
      libfreenect2::Frame*, 
      libfreenect2::Frame*, 
      libfreenect2::Frame*,
      libfreenect2::Registration*, 
      libfreenect2::Frame*, 
      libfreenect2::Frame*)>);

}
}

#endif  // KINECT_TEST_FREENECT_H_INCLUDED