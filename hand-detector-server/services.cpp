#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <strstream>
#include <ctime>

#include "json/json.h"

#include "vision/hand-detection.hpp"
#include "vision/screen-detection.hpp"
#include "vision/cv-debugging.hpp"
#include "vision/cv-helper.hpp"
#include "camera.hpp"
#include "options.hpp"
#include "services.hpp"

using std::string;
using std::vector;
using cv::Mat;
using cv::imread;
using Json::Value;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void saveHandInput(string path, Mat &rgb, Mat &depth, Mat &depthBg, Mat &proj)
{

  if (path == "") {
    std::time_t t = std::time(0);
    path = std::to_string(t) + ".yml";
  }

  cv::FileStorage fs(path, cv::FileStorage::WRITE);

  fs << "rgb" << rgb
     << "depth" << depth
     << "depthBackground" << depthBg
     << "projection" << proj;

  fs.release();
}

Mat depthDiff(Value msg, Mat &image, Mat &depthImage, Mat &depthBackground)
{
  Mat diff(depthImage.size(), CV_32F);
  absdiff(depthImage, depthBackground, diff);
  cv::normalize(diff, diff);
  auto depthThreshold = msg["data"].get("depthThreshold", 0).asFloat();
  cv::threshold(diff, diff, depthThreshold, 1.0f, cv::THRESH_BINARY);
  return diff;
}

// void debugDepth(Value msg, Mat &image, Mat &depthImage, Mat &depthBackground, Mat &proj)
// {
//   Mat diff(depthImage.size(), CV_32F);
//   absdiff(depthImage, depthBackground, diff);
//   cv::normalize(diff, diff);
//   auto depthThreshold = msg["data"].get("depthThreshold", 0).asFloat();
//   Mat diff2;
//   cv::threshold(diff, diff2, depthThreshold, 1.0f, cv::THRESH_BINARY);
//   saveHandInput("test.yml", image, depthImage, depthBackground, proj);
//   imshow("depth diff 1", diff);
//   imshow("depth diff 2", diff2);
//   cv::waitKey(30);
// }

void transformFrame(Mat &input, Mat &output, cv::Size &tfmedSize, Mat &proj)
{
  output.create(tfmedSize, input.type());
  cv::warpPerspective(input, output, proj, tfmedSize);
}

void answerWithError(Server &server, Value &msg, string errMessage)
{
  std::cout << errMessage << std::endl;
  Value answer;
  answer["data"]["error"] = true;
  answer["data"]["message"] = errMessage;
  server->answer(msg, answer);
}

vision::cam::CameraPtr getVideoCaptureDev(Value &msg)
{
  std::string videoDevName = msg["data"].get("deviceNo", "").asString();
  try {
    return vision::cam::getCamera(std::stoi(videoDevName));
  } catch(const std::exception& e) {
    return vision::cam::getCamera(videoDevName);
  }
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

bool uploadedOrCapturedImage(
  Server &server, string &sender, Value &msg,
  Mat &image, Mat &depthImage, Mat &depthBackgroundImage)
{
  // 1. read depth
  auto depthFile = msg["data"].get("depthFile", "").asString();
  if (depthFile != "") {
    std::cout << "Reading depth file: " << depthFile << std::endl;
    cv::FileStorage fs(depthFile, cv::FileStorage::READ);
    fs["depth"] >> depthImage;
  }

  auto depthBackgroundFile = msg["data"].get("depthBackgroundFile", "").asString();
  if (depthBackgroundFile != "") {
    std::cout << "Reading depth background file: " << depthBackgroundFile << std::endl;
    cv::FileStorage fs(depthBackgroundFile, cv::FileStorage::READ);
    fs["depth"] >> depthBackgroundImage;
  }

  // 2. image uploaded?
  if (uploadedDataToMat(server, sender, msg, image)) return true;

  // 3. capture image
  getVideoCaptureDev(msg)->readWithDepth(image, depthImage);

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
    if (maxWidth > 0 && maxHeight > 0)
      cvhelper::resizeToFit(depthFrame, depthFrame, maxWidth, maxHeight);
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
  string depthFile = msg["data"].get("depthFile", "").asString();
  auto cam = getVideoCaptureDev(msg);
  Mat frame, depthFrame;
  server->answer(msg, (string)"OK");
  readFrameAndSend(nFrames, maxWidth, maxHeight, cam, frame, depthFrame, server, sender);
  if (depthFile != "") {
    cv::FileStorage fs(depthFile, cv::FileStorage::WRITE);
    fs << "depth" << depthFrame;
    fs.release();
  }
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

  vision::screen::Options opts = screenOptions(msg["data"]);

  Mat image, depthImage, depthBackgroundImage;
  uploadedOrCapturedImage(server, sender, msg, image, depthImage, depthBackgroundImage);

  if (maxWidth > 0 && maxHeight > 0) {
    cvhelper::resizeToFit(image, image, maxWidth, maxHeight);
    if (!depthImage.empty()) cvhelper::resizeToFit(depthImage, depthImage, maxWidth, maxHeight);
    if (!depthBackgroundImage.empty()) cvhelper::resizeToFit(depthBackgroundImage, depthBackgroundImage, maxWidth, maxHeight);
  }

  sendMat(image, server, sender);

  image.convertTo(image, CV_8UC1);
  auto corners = vision::screen::cornersOfLargestRect(image, opts, debug).asVector();
  if (debug) {
    Mat recorded = cvdbg::getAndClearRecordedImages();
    cv::imshow("debug2", recorded);
    cv::waitKey(30);
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

  vision::screen::Options opts = screenOptions(msg["data"]);

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

  Mat image, depthImage, depthBackgroundImage;
  uploadedOrCapturedImage(server, sender, msg, image, depthImage, depthBackgroundImage);

  auto maxWidth = msg["data"].get("maxWidth", 0).asInt();
  auto maxHeight = msg["data"].get("maxHeight", 0).asInt();
  if (maxWidth > 0 && maxHeight > 0) {
    cvhelper::resizeToFit(image, image, maxWidth, maxHeight);
    if (!depthImage.empty()) cvhelper::resizeToFit(depthImage, depthImage, maxWidth, maxHeight);
    if (!depthBackgroundImage.empty()) cvhelper::resizeToFit(depthBackgroundImage, depthBackgroundImage, maxWidth, maxHeight);
  }

  string saveAs = msg["data"].get("saveAs", "").asString();
  if (saveAs != "") {
    saveHandInput(saveAs, image, depthImage, depthBackgroundImage.empty() ? depthImage : depthBackgroundImage, proj);
  }

  bool debug = false;
  vision::screen::Options opts = screenOptions(msg["data"]);
  Mat projected = vision::screen::applyScreenProjection(image, proj, image.size(), opts, debug);
  if (debug) {
    Mat recorded = cvdbg::getAndClearRecordedImages();
    imshow("debug-screenTransform", recorded);
    cv::waitKey(30);
  }

  sendMat(projected, server, sender);

  server->answer(msg, (string)"OK");
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void recognizeHand(
  Value &msg,
  Mat &in, Mat &depth, Mat &depthBackground, Mat &proj, Mat &out,
  vision::hand::FrameWithHands &handData,
  int maxWidth, int maxHeight,
  vision::hand::Options &opts,
  bool record = false, bool debug = false)
{
  if (maxWidth > 0 && maxHeight > 0) {
    cvhelper::resizeToFit(in, in, maxWidth, maxHeight);
    if (!depth.empty()) cvhelper::resizeToFit(depth, depth, maxWidth, maxHeight);
    if (!depthBackground.empty()) cvhelper::resizeToFit(depthBackground, depthBackground, maxWidth, maxHeight);
  }

  if (record) {
    saveHandInput("", in, depth, depthBackground, proj);
  }

  Mat output;
  cv::Size tfmedSize = in.size();
  transformFrame(in, output, tfmedSize, proj);
  if (depth.empty()) depth = Mat::zeros(in.size(), CV_32F);
  transformFrame(depth, depth, tfmedSize, proj);
  if (depthBackground.empty()) depthBackground = Mat::zeros(in.size(), CV_32F);
  transformFrame(depthBackground, depthBackground, tfmedSize, proj);

  Mat diff = depthDiff(msg, in, depth, depthBackground);
  vision::hand::processFrame(output, depth, depthBackground, output, handData, opts, debug);

  // debugging...
  if (debug) {
    // std::cout << vision::hand::frameWithHandsToJSONString(handData) << std::endl;

    Mat diffu = diff.clone();
    diffu *= 255;
    diffu.convertTo(diffu, CV_8UC3);
    cvdbg::recordImage(diffu, "diff");
    // debugDepth(msg, in, depth, depthBackground, proj);
    Mat recorded = cvdbg::getAndClearRecordedImages();
    // recorded.copyTo(out);
    cvhelper::resizeToFit(recorded, out, maxWidth, maxWidth);
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
  auto playbackFile = msg["data"].get("playbackFile", "").asString();
  auto backgroundFile = msg["data"].get("backgroundFile", "").asString();

  bool record = false;
  Mat image, depthImage, depthBackground, proj = Mat::eye(3,3,CV_32F);

  if (playbackFile != "") {
    std::cout << "playbackFile file: " << playbackFile << std::endl;
    cv::FileStorage fs(playbackFile, cv::FileStorage::READ);
    fs["rgb"] >> image;
    fs["projection"] >> proj;
    fs["depth"] >> depthImage;
    fs["depthBackground"] >> depthBackground;
  } else if (backgroundFile != "") {
    std::cout << "backgroundFile file: " << backgroundFile << std::endl;
    cv::FileStorage fs(backgroundFile, cv::FileStorage::READ);
    fs["projection"] >> proj;
    fs["depthBackground"] >> depthBackground;
    uploadedOrCapturedImage(server, sender, msg, image, depthImage, depthBackground);
    record = true;
  } else {
    Value projection = msg["data"]["projection"];
    if (projection.isArray() && projection.size() == 3*3) {
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          proj.at<float>(i, j) = projection[(i*3)+j].asFloat();
    }
    uploadedOrCapturedImage(server, sender, msg, image, depthImage, depthBackground);
    record = true;
  }


  if (msg["data"].isMember("record")) record = msg["data"]["record"].asBool();
  bool debug = msg["data"]["debug"].asBool();
  vision::hand::Options opts = handOptions(msg["data"]);
  vision::hand::FrameWithHands handData;
  Mat recorded;
  recognizeHand(msg, image, depthImage, depthBackground, proj, recorded, handData, maxWidth, maxHeight, opts, record, debug);

  sendMat(recorded, server, sender);

  server->answer(msg, (string)"OK");
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

std::map<std::string, bool> handDetectionActivities;

void runHandDetectionProcessFor(
  string &target, Server &server, Value &msg,
  Mat &frame, Mat &depthFrame, Mat &depthBackground, Mat &proj, vision::cam::CameraPtr &dev,
  uint maxWidth, uint maxHeight,
  vision::hand::Options &opts,
  bool record, bool debug = false)
{
  // still running?
  if (!handDetectionActivities[target]) {
    std::cout << "stopping hand detection for " << target << std::endl;
    return;
  }

  try {
    dev->readWithDepth(frame, depthFrame);

    vision::hand::FrameWithHands handData;
    Mat recorded;
    recognizeHand(msg, frame, depthFrame, depthBackground, proj, recorded, handData, maxWidth, maxHeight, opts, record, debug);

    sendMat(recorded, server, target);
    cv::waitKey(30);
    // sendMat(frame, server, target);
    server->setTimer(10, bind(runHandDetectionProcessFor,
      target, server, msg,
      frame, depthFrame, depthBackground, proj, dev,
      maxWidth, maxHeight,
      opts, record, debug));
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
  bool record = msg["data"]["record"].asBool();
  auto maxWidth = msg["data"].get("maxWidth", 0).asInt();
  auto maxHeight = msg["data"].get("maxHeight", 0).asInt();
  auto backgroundFile = msg["data"].get("backgroundFile", "").asString();
  auto cam = getVideoCaptureDev(msg);
  Mat image, depthImage, depthBackground;

  handDetectionActivities[sender] = true;

  Mat proj = Mat::eye(3,3,CV_32F);
  if (backgroundFile != "") {
    std::cout << "backgroundFile file: " << backgroundFile << std::endl;
    cv::FileStorage fs(backgroundFile, cv::FileStorage::READ);
    fs["projection"] >> proj;
    fs["depthBackground"] >> depthBackground;
    uploadedOrCapturedImage(server, sender, msg, image, depthImage, depthBackground);
  } else {
    Value projection = msg["data"]["projection"];
    if (projection.isArray() && projection.size() == 3*3) {
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          proj.at<float>(i, j) = projection[(i*3)+j].asFloat();
    }
    uploadedOrCapturedImage(server, sender, msg, image, depthImage, depthBackground);
  }

  vision::hand::Options opts = handOptions(msg["data"]);

  runHandDetectionProcessFor(sender, server, msg, image, depthImage, depthBackground, proj, cam, maxWidth, maxHeight, opts, record, debug);

  server->answer(msg, (string)"OK");
}

}
}
