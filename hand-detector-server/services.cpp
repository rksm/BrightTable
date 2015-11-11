#include <iostream>
#include <string>

#include "vision/hand-detection.hpp"
#include "vision/screen-detection.hpp"
#include "vision/cv-helper.hpp"
#include "json/json.h"
#include "services.hpp"

using std::string;
using std::vector;
using cv::Mat;
using cv::imread;
using Json::Value;

void addErrorMessage(Value &msg, string errMessage)
{
  msg["data"]["error"] = true;
  msg["data"]["message"] = errMessage;
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

std::mutex capsMutex;
typedef std::shared_ptr<cv::VideoCapture> CaptureDev;
typedef std::map<int, CaptureDev> CaptureDevs;
CaptureDevs caps;
CaptureDev getVideoCaptureDev(int devNo)
{
  auto capIt = caps.find(devNo);
  if (capIt != caps.end()) return capIt->second;
  
  auto cap = std::make_shared<cv::VideoCapture>(cv::VideoCapture(devNo));
  capsMutex.lock();
  caps.insert({devNo, cap});
  capsMutex.unlock();
  return cap;
}

namespace handdetection {
namespace server {
  
void readFrameAndSend(
  uint repeat,
  uint maxWidth,
  uint maxHeight,
  CaptureDev dev,
  Mat &frame,
  Server &server,
  string &target,
  vector<int> &params,
  vector<uchar> &buffer)
{
  if (repeat == 0) return;

  try {
    dev->read(frame);
    // resize(frame, frame, cv::Size(), 0.3, 0.3);
    if (maxWidth > 0 && maxHeight > 0)
      cvhelper::resizeToFit(frame, frame, maxWidth, maxHeight);
    imencode(".jpg", frame, buffer, params);
    server->sendBinary(target, &buffer[0], buffer.size());
    server->setTimer(10, bind(readFrameAndSend, repeat - 1, maxWidth, maxHeight, dev, frame, server, target, params, buffer));
  } catch (const std::exception& e) {
    std::cout << "error in readFrameAndSend: " << e.what() << std::endl;
  }
}


void captureCameraService(Value msg, Server server)
{
  auto sender = msg.get("sender", "").asString();

  auto maxWidth = msg["data"].get("maxWidth", 0).asInt();
  auto maxHeight = msg["data"].get("maxHeight", 0).asInt();
  auto nFrames = msg["data"].get("nFrames", 1).asInt();
  auto videoDevNo = msg["data"].get("deviceNo", 0).asInt();
  auto cap = getVideoCaptureDev(videoDevNo);
  
  Mat frame;
  vector<int> params = vector<int>(2);
  params[0]=CV_IMWRITE_JPEG_QUALITY;
  params[1]=95;
  vector<uchar> buffer;
  
  readFrameAndSend(nFrames, maxWidth, maxHeight, cap, frame, server, sender, params, buffer);
}


}
}
