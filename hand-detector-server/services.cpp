#include <iostream>
#include <string>
#include <strstream>

#include "camera.hpp"

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

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

int thresholdType(std::string name)
{
  if (name == "CV_THRESH_BINARY")          return CV_THRESH_BINARY;
  else if (name == "CV_THRESH_BINARY_INV") return CV_THRESH_BINARY_INV;
  else if (name == "CV_THRESH_TRUNC")      return CV_THRESH_TRUNC;
  else if (name == "CV_THRESH_TOZERO")     return CV_THRESH_TOZERO;
  else if (name == "CV_THRESH_TOZERO_INV") return CV_THRESH_TOZERO_INV;
  else if (name == "CV_THRESH_MASK")       return CV_THRESH_MASK;
  else if (name == "CV_THRESH_OTSU")       return CV_THRESH_OTSU;
  else                                     return CV_THRESH_BINARY;
}

void quadOptions(Value &data, vision::quad::Options &opts)
{
  if (data.isMember("minAngleOfIntersectingLines")) opts.minAngleOfIntersectingLines = data["minAngleOfIntersectingLines"].asFloat();
  if (data.isMember("maxAngleOfIntersectingLines")) opts.maxAngleOfIntersectingLines = data["maxAngleOfIntersectingLines"].asFloat();
}

void screenOptions(Value &data, vision::screen::Options &opts)
{
  if (data.isMember("blurIntensity"))      opts.blurIntensity      = data["blurIntensity"].asInt();
  if (data.isMember("minThreshold"))       opts.minThreshold       = data["minThreshold"].asInt();
  if (data.isMember("maxThreshold"))       opts.maxThreshold       = data["maxThreshold"].asInt();
  if (data.isMember("thresholdType"))      opts.thresholdType      = thresholdType(data["thresholdType"].asString());
  if (data.isMember("houghRho"))           opts.houghRho           = data["houghRho"].asInt();
  if (data.isMember("houghTheta"))         opts.houghTheta         = data["houghTheta"].asInt();
  if (data.isMember("houghMinLineLength")) opts.houghMinLineLength = data["houghMinLineLength"].asInt();
  if (data.isMember("houghMinLineGap"))    opts.houghMinLineGap    = data["houghMinLineGap"].asInt();
  if (data.isMember("quadOptions")) {
    quadOptions(data["quadOptions"], opts.quadOptions);
  }
}

void handOptions(Value &data, vision::hand::Options &opts)
{
  if (data.isMember("fingerTipWidth"))   opts.fingerTipWidth   = data["fingerTipWidth"].asInt();
  if (data.isMember("blurIntensity"))    opts.blurIntensity    = data["blurIntensity"].asInt();
  if (data.isMember("thresholdMin"))     opts.thresholdMin     = data["thresholdMin"].asInt();
  if (data.isMember("thresholdMax"))     opts.thresholdMax     = data["thresholdMax"].asInt();
  if (data.isMember("thresholdType"))    opts.thresholdType    = thresholdType(data["thresholdType"].asString());
  if (data.isMember("dilateIterations")) opts.dilateIterations = data["dilateIterations"].asInt();
  if (data.isMember("cropWidth"))        opts.cropWidth        = data["cropWidth"].asInt();
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void answerWithError(Server &server, Value &msg, string errMessage)
{
  std::cout << errMessage << std::endl;
  Value answer;
  answer["data"]["error"] = true;
  answer["data"]["message"] = errMessage;
  server->answer(msg, answer);
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

vision::cam::CameraPtr getVideoCaptureDev(Value &msg)
{
  std::string videoDevName = msg["data"].get("deviceNo", "").asString();
  try {
    return vision::cam::getCamera(std::stoi(videoDevName));
  } catch(const std::exception& e) {
    return vision::cam::getCamera(videoDevName);
  }
}

namespace handdetection {
namespace server {

bool uploadedDataToMat(Server &server, string &sender, Value &msg, Mat &image)
{
  auto uploaded = server->getUploadedBinaryDataOf(sender);
  server->clearUploadedBinaryDataOf(sender);
  if (uploaded.empty()) { image = Mat(); return false; }

  vector<uchar> data(uploaded[0]->data, uploaded[0]->data + uploaded[0]->size);
  image = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
  return true;
}

bool uploadedOrCapturedImage(Server &server, string &sender, Value &msg, Mat &image)
{
  if (uploadedDataToMat(server, sender, msg, image)) return true;

  auto maxWidth = msg["data"].get("maxWidth", 0).asInt();
  auto maxHeight = msg["data"].get("maxHeight", 0).asInt();
  getVideoCaptureDev(msg)->read(image);
  if (maxWidth > 0 && maxHeight > 0)
    cvhelper::resizeToFit(image, image, maxWidth, maxHeight);
  return false;
}

void sendMat(Mat &mat, Server &server, string &target)
{
  auto params = vector<int>(2);
  params[0] = CV_IMWRITE_JPEG_QUALITY;
  params[1] = 75;
  vector<uchar> buffer;
  imencode(".jpg", mat, buffer, params);
  server->sendBinary(target, &buffer[0], buffer.size());
}

void readFrameAndSend(
  uint &repeat,
  uint &maxWidth,
  uint &maxHeight,
  vision::cam::CameraPtr dev,
  Mat &frame,
  Mat &depthFrame,
  Server &server,
  string &target)
{
  if (repeat == 0) return;

  try {
    dev->readWithDepth(frame, depthFrame);
    // resize(frame, frame, cv::Size(), 0.3, 0.3);
    if (maxWidth > 0 && maxHeight > 0)
      cvhelper::resizeToFit(frame, frame, maxWidth, maxHeight);
    sendMat(frame, server, target);
    sendMat(depthFrame, server, target);
    server->setTimer(10, bind(readFrameAndSend, repeat - 1, maxWidth, maxHeight, dev, frame, depthFrame, server, target));
  } catch (const std::exception& e) {
    std::cout << "error in readFrameAndSend: " << e.what() << std::endl;
  }
}


void captureCameraService(Value msg, Server server)
{
  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }
  uint maxWidth = msg["data"].get("maxWidth", 0).asInt();
  uint maxHeight = msg["data"].get("maxHeight", 0).asInt();
  uint nFrames = msg["data"].get("nFrames", 1).asInt();
  auto cam = getVideoCaptureDev(msg);
  Mat frame, depthFrame;
  server->answer(msg, (string)"OK");
  readFrameAndSend(nFrames, maxWidth, maxHeight, cam, frame, depthFrame, server, sender);
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void uploadImageService(Value msg, Server server)
{
  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }
  auto path = msg["data"].get("path", "").asString();
  if (path == "") { answerWithError(server, msg, "no path"); return; }
  int h = msg["data"].get("height", 0).asInt(),
      w = msg["data"].get("width", 0).asInt();

  // auto maxWidth = msg["data"].get("path", "").asString();

  Mat uploaded;
  if (!uploadedDataToMat(server, sender, msg, uploaded)) {
    answerWithError(server, msg, "no data uploaded");
    return;
  }

  imshow("out", uploaded);

  server->answer(msg, (string)"OK");
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void recognizeScreenCornersService(Value msg, Server server)
{
  // guess the corners of the largest quad area in tge uploaded image. Used for
  // screenCornersTransform

  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }
  auto maxWidth = msg["data"].get("maxWidth", 0).asInt();
  auto maxHeight = msg["data"].get("maxHeight", 0).asInt();
  bool debug = msg["data"]["debug"].asBool();

  vision::screen::Options opts;
  screenOptions(msg["data"], opts);

  Mat image;
  if (!uploadedOrCapturedImage(server, sender, msg, image))
    sendMat(image, server, sender);
  sendMat(image, server, sender);

  auto corners = vision::screen::cornersOfLargestRect(image, opts, debug).asVector();
  if (debug) {
    // Mat recorded = cvhelper::getAndClearRecordedImages();
    // if (maxWidth > 0 && maxHeight > 0)
    //   cvhelper::resizeToFit(recorded, recorded, maxWidth, maxHeight);
    // sendMat(recorded, server, sender);
  }

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
  answer["data"]["size"]["width"] = image.size().width;
  answer["data"]["size"]["height"] = image.size().height;
  server->answer(msg, answer);
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void screenCornersTransform(Value msg, Server server)
{
  // givent four corners and a "screen area" (width, height), produce a 3x3
  // matrix to crop transform the area identified byt the corners into a rectangle
  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }
  float tlX = msg["data"]["corners"][0]["x"].asFloat(),
        tlY = msg["data"]["corners"][0]["y"].asFloat(),
        trX = msg["data"]["corners"][1]["x"].asFloat(),
        trY = msg["data"]["corners"][1]["y"].asFloat(),
        brX = msg["data"]["corners"][2]["x"].asFloat(),
        brY = msg["data"]["corners"][2]["y"].asFloat(),
        blX = msg["data"]["corners"][3]["x"].asFloat(),
        blY = msg["data"]["corners"][3]["y"].asFloat();
  vision::quad::Corners corners(tlX,tlY,trX,trY,brX,brY,blX,blY);

  float width = msg["data"]["size"].get("width", 0).asFloat(),
        height = msg["data"]["size"].get("height", 0).asFloat();
  cv::Rect bounds(0,0, width, height);

  vision::screen::Options opts;
  screenOptions(msg["data"], opts);

  Mat projection = vision::quad::cornerTransform(corners, bounds, opts.quadOptions);
  projection.convertTo(projection, CV_32F);

  Value answer;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      answer["data"]["projection"][(i*3)+j] = projection.at<float>(i, j);

  server->answer(msg, answer);
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void screenTransform(Value msg, Server server)
{
  // expects a 3x3 projection matrix, e.g. from screenCornersTransform and
  // transforms the uploaded image with it
  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }

  Value projection = msg["data"]["projection"];

  vector<float> projArr;
  for (auto it = projection.begin(); it != projection.end(); it++)
    projArr.push_back((*it).asFloat());

  Mat proj = Mat::eye(3,3,CV_32F);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      proj.at<float>(i, j) = projArr[(i*3)+j];

  Mat image;
  uploadedOrCapturedImage(server, sender, msg, image);

  bool debug = false;
  vision::screen::Options opts;
  screenOptions(msg["data"], opts);
  Mat projected = vision::screen::applyScreenProjection(image, proj, image.size(), opts, debug);
  if (debug) {
    imshow("image", image);
    Mat recorded = cvhelper::getAndClearRecordedImages();
    imshow("debug", recorded);
  }

  sendMat(projected, server, sender);

  server->answer(msg, (string)"OK");
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

Mat transformFrame(Mat &input, cv::Size &tfmedSize, Mat &proj)
{
  cvhelper::resizeToFit(input, input, 700, 700);
  cv::Mat output = Mat(tfmedSize, CV_8UC3);
  cv::warpPerspective(input, output, proj, output.size());
  return output;
}

void recognizeHand(
  Mat &in, Mat &proj, Mat &out,
  vision::hand::FrameWithHands &handData,
  int maxWidth, int maxHeight,
  vision::hand::Options &opts,
  bool debug = false)
{

  if (maxWidth > 0 && maxHeight > 0)
    cvhelper::resizeToFit(in, in, maxWidth, maxHeight);

  // cv::Size tfmedSize(820, 432);
  cv::Size tfmedSize = in.size();

  // step 1: apply screen transform
  // std::cout << "hand detection uses projection " << proj << " into " << tfmedSize << std::endl;

  // Mat output = in;
  Mat output = transformFrame(in, tfmedSize, proj);
  // imshow("1", output);

  // step 2: find hand + finger
  vision::hand::processFrame(output, output, handData, opts, debug);
  // imshow("2", output);

  // debugging...
  if (debug) {
    // std::cout << vision::hand::frameWithHandsToJSONString(handData) << std::endl;
    Mat recorded = cvhelper::getAndClearRecordedImages();
    cvhelper::resizeToFit(recorded, out, maxWidth,maxWidth);
  } else {
    out = in;
  }
}

void handDetection(Value msg, Server server)
{
  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }

  auto maxWidth = msg["data"].get("maxWidth", 0).asInt();
  auto maxHeight = msg["data"].get("maxHeight", 0).asInt();

  Mat proj = Mat::eye(3,3,CV_32F);
  Value projection = msg["data"]["projection"];
  if (projection.isArray() && projection.size() == 3*3) {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        proj.at<float>(i, j) = projection[(i*3)+j].asFloat();
  }

  Mat uploaded;
  if (!uploadedDataToMat(server, sender, msg, uploaded)) {
    answerWithError(server, msg, "no data uploaded"); return; }

  bool debug = msg["data"]["debug"].asBool();
  vision::hand::Options opts;
  handOptions(msg["data"], opts);
  vision::hand::FrameWithHands handData;
  Mat recorded;
  recognizeHand(uploaded, proj, recorded, handData, maxWidth, maxHeight, opts, debug);

  sendMat(recorded, server, sender);

  server->answer(msg, (string)"OK");
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

std::map<std::string, bool> handDetectionActivities;

void runHandDetectionProcessFor(
  string &target, Server &server,
  Mat &frame, Mat &proj, vision::cam::CameraPtr &dev,
  uint maxWidth, uint maxHeight,
  vision::hand::Options &opts,
  bool debug = false)
{
  // still running?
  if (!handDetectionActivities[target]) {
    std::cout << "stopping hand detection for " << target << std::endl;
    return;
  }

  try {
    dev->read(frame);
    vision::hand::FrameWithHands handData;
    Mat recorded;
    recognizeHand(frame, proj, recorded, handData, maxWidth, maxHeight, opts, debug);
    sendMat(recorded, server, target);
    server->setTimer(10, bind(runHandDetectionProcessFor,
      target, server,
      frame, proj, dev,
      maxWidth, maxHeight,
      opts, debug));
  } catch (const std::exception& e) {
    std::cout << "error in runHandDetectionProcessFor: " << e.what() << std::endl;
  }
}

void handDetectionStreamStop(Value msg, Server server)
{
  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }
  handDetectionActivities[sender] = false;
}

void handDetectionStreamStart(Value msg, Server server)
{
  // expects a 3x3 projection matrix, e.g. from screenCornersTransform and
  // transforms the uploaded image with it
  auto sender = msg.get("sender", "").asString();
  if (sender == "") { answerWithError(server, msg, "no sender"); return; }

  bool debug = msg["data"]["debug"].asBool();
  auto maxWidth = msg["data"].get("maxWidth", 0).asInt();
  auto maxHeight = msg["data"].get("maxHeight", 0).asInt();
  auto cam = getVideoCaptureDev(msg);
  Mat frame;

  handDetectionActivities[sender] = true;

  Mat proj = Mat::eye(3,3,CV_32F);
  Value projection = msg["data"]["projection"];
  if (projection.size() == 3*3) {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        proj.at<float>(i, j) = projection[(i*3)+j].asFloat();
  }

  vision::hand::Options opts;
  handOptions(msg["data"], opts);

  runHandDetectionProcessFor(sender, server, frame, proj, cam, maxWidth, maxHeight, opts, debug);

  server->answer(msg, (string)"OK");
}

}
}
