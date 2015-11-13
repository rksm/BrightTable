#include <iostream>
#include <string>
#include <strstream>

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

void answerWithError(Server &server, Value &msg, string errMessage)
{
  std::cout << errMessage << std::endl;
  Value answer;
  answer["data"]["error"] = true;
  answer["data"]["message"] = errMessage;
  server->answer(msg, answer);
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

Mat uploadedDataToMat(Server server, string sender, Value msg)
{
  auto uploaded = server->getUploadedBinaryDataOf(sender);
  server->clearUploadedBinaryDataOf(sender);
  if (uploaded.empty()) {
    answerWithError(server, msg, "no data uploaded");
    return Mat();
  }

  vector<uchar> data(uploaded[0]->data, uploaded[0]->data + uploaded[0]->size);
  Mat image = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
  return image;
}

void sendMat(Mat &mat, Server &server, string &target)
{
  auto params = vector<int>(2);
  params[0] = CV_IMWRITE_JPEG_QUALITY;
  params[1] = 95;
  vector<uchar> buffer;
  imencode(".jpg", mat, buffer, params);
  server->sendBinary(target, &buffer[0], buffer.size());
}

void readFrameAndSend(
  uint repeat,
  uint maxWidth,
  uint maxHeight,
  CaptureDev dev,
  Mat &frame,
  Server &server,
  string &target)
{
  if (repeat == 0) return;

  try {
    dev->read(frame);
    // resize(frame, frame, cv::Size(), 0.3, 0.3);
    if (maxWidth > 0 && maxHeight > 0)
      cvhelper::resizeToFit(frame, frame, maxWidth, maxHeight);
    sendMat(frame, server, target);
    server->setTimer(10, bind(readFrameAndSend, repeat - 1, maxWidth, maxHeight, dev, frame, server, target));
  } catch (const std::exception& e) {
    std::cout << "error in readFrameAndSend: " << e.what() << std::endl;
  }
}


void captureCameraService(Value msg, Server server)
{
  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }
  auto maxWidth = msg["data"].get("maxWidth", 0).asInt();
  auto maxHeight = msg["data"].get("maxHeight", 0).asInt();
  auto nFrames = msg["data"].get("nFrames", 1).asInt();
  auto videoDevNo = msg["data"].get("deviceNo", 0).asInt();
  auto cap = getVideoCaptureDev(videoDevNo);
  Mat frame;
  server->answer(msg, (string)"OK");
  readFrameAndSend(nFrames, maxWidth, maxHeight, cap, frame, server, sender);
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void uploadImageService(Value msg, Server server)
{
  std::cout << msg << std::endl;

  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }
  auto path = msg["data"].get("path", "").asString();
  if (path == "") { answerWithError(server, msg, "no path"); return; }
  int h = msg["data"].get("height", 0).asInt(),
      w = msg["data"].get("width", 0).asInt();

  // auto maxWidth = msg["data"].get("path", "").asString();

  Mat uploaded = uploadedDataToMat(server, sender, msg);
  if (uploaded.empty()) return;

  imshow("out", uploaded);

  server->answer(msg, (string)"OK");
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void recognizeScreenCornersService(Value msg, Server server)
{
  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }

  Mat uploaded = uploadedDataToMat(server, sender, msg);
  if (uploaded.empty()) return;

  auto corners = cornersOfLargestRect(uploaded).asVector();
  if (corners.size() != 4) {
    answerWithError(server, msg, "did not find 4 corners but " + std::to_string(corners.size()));
    return;
  }

  Value answer;
  for (int i = 0; i < 4; i++)
  {
    answer["data"]["corners"][i]["x"] = corners[i].x;
    answer["data"]["corners"][i]["y"] = corners[i].y;
  }
  answer["data"]["size"]["width"] = uploaded.size().width;
  answer["data"]["size"]["height"] = uploaded.size().height;
  server->answer(msg, answer);
}

void screenCornersTransform(Value msg, Server server)
{
  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }
  float tlX = msg["data"]["corners"][0].asFloat();
  float tlY = msg["data"]["corners"][1].asFloat();
  float trX = msg["data"]["corners"][2].asFloat();
  float trY = msg["data"]["corners"][3].asFloat();
  float brX = msg["data"]["corners"][4].asFloat();
  float brY = msg["data"]["corners"][5].asFloat();
  float blX = msg["data"]["corners"][6].asFloat();
  float blY = msg["data"]["corners"][7].asFloat();
  Corners corners(tlX,tlY,trX,trY,brX,brY,blX,blY);

  float width = msg["data"]["size"].get("width", 0).asFloat();
  float height = msg["data"]["size"].get("height", 0).asFloat();
  cv::Rect bounds(0,0, width, height);

  Mat projection = cornerTransform(corners, bounds);
  std::cout << projection << std::endl;

  Mat uploaded = uploadedDataToMat(server, sender, msg);
  if (uploaded.empty()) return;

  Mat projected = applyScreenProjection(uploaded, projection, cv::Size(width, height));
  sendMat(projected, server, sender);

  Value answer;
  std::vector<float> array;
  array.assign((float*)projection.datastart, (float*)projection.dataend);
  for (int i = 0; i < array.size(); i++)
    answer["data"]["projection"][i] = array[i];
  server->answer(msg, answer);
}

}
}
