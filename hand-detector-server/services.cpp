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

template<typename T>
void sortedValues(Mat &in, vector<T> &out)
{
  auto size = in.size();
  for (int i = 0; i < size.height; i++)
    for (int j = 0; j < size.width; j++)
      out[i*size.width + j] = in.at<T>(i, j);
  std::sort(out.begin(), out.end());
}

template<typename T>
std::pair<T,T> percentiles(Mat &in, float percentile=5.0f) {
  
  size_t n = in.cols * in.rows;
  vector<T> sorted(n);
  sortedValues(in, sorted);
  
  size_t idx = std::ceil(percentile/100.0f * (float)n);
  auto low = sorted[idx];
  auto high = sorted[n - idx];

  std::cout << "percentile: " << percentile
            << " n: " << n
            << " low: " << low
            << " high: " << high
            << std::endl;

  return std::make_pair(low, high);
}

void convertToProperGrayscale(Mat &grayMat, float percentile=5.0f)
{
  // float high = 1.0f, low = 0.0f;
  // if (percentile != 0.0f) {
    auto perc = percentiles<float>(grayMat, percentile);
    float high = perc.second, low = perc.first;
  // }

  for (int i = 0; i < grayMat.rows; i++)
    for (int j = 0; j < grayMat.cols; j++)
    {
      auto val = grayMat.at<float>(i,j);
      grayMat.at<float>(i,j) = std::max(0.0f, std::min(1.0f, 1.0f/(high-low) * (val-low)));
    }
  // float multiplier = 1 / perc.second;
  // grayMat = (grayMat - perc.first) * multiplier;
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

bool uploadedOrCapturedImage(Server &server, string &sender, Value &msg, Mat &image, Mat &depthImage, Mat &depthBackgroundImage)
{

  auto maxWidth = msg["data"].get("maxWidth", 0).asInt();
  auto maxHeight = msg["data"].get("maxHeight", 0).asInt();

  // 1. read depth
  auto depthFile = msg["data"].get("depthFile", "").asString();
  if (depthFile != "") {
    std::cout << "Reading depth file: " << depthFile << std::endl;
    cv::FileStorage fs(depthFile, cv::FileStorage::READ);
    fs["depth"] >> depthImage;
    if (maxWidth > 0 && maxHeight > 0) cvhelper::resizeToFit(depthImage, depthImage, maxWidth, maxHeight);
  }

  auto depthBackgroundFile = msg["data"].get("depthBackgroundFile", "").asString();
  if (depthBackgroundFile != "") {
    std::cout << "Reading depth background file: " << depthBackgroundFile << std::endl;
    cv::FileStorage fs(depthBackgroundFile, cv::FileStorage::READ);
    fs["depth"] >> depthBackgroundImage;
    if (maxWidth > 0 && maxHeight > 0) cvhelper::resizeToFit(depthBackgroundImage, depthBackgroundImage, maxWidth, maxHeight);
  }

  // 2. image uploaded?
  if (uploadedDataToMat(server, sender, msg, image)) return true;

  // 3. capture image
  getVideoCaptureDev(msg)->readWithDepth(image, depthImage);
  if (maxWidth > 0 && maxHeight > 0) {
    cvhelper::resizeToFit(image, image, maxWidth, maxHeight);
    cvhelper::resizeToFit(depthImage, depthImage, maxWidth, maxHeight);
  }

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

  vision::screen::Options opts;
  screenOptions(msg["data"], opts);

  Mat image, depthImage, depthBackgroundImage;
  if (!uploadedOrCapturedImage(server, sender, msg, image, depthImage, depthBackgroundImage))
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

  Mat image, depthImage, depthBackgroundImage;
  uploadedOrCapturedImage(server, sender, msg, image, depthImage, depthBackgroundImage);

  bool debug = false;
  vision::screen::Options opts;
  screenOptions(msg["data"], opts);
  Mat projected = vision::screen::applyScreenProjection(image, proj, image.size(), opts, debug);
  if (debug) {
    Mat recorded = cvdbg::getAndClearRecordedImages();
  }

  sendMat(projected, server, sender);

  server->answer(msg, (string)"OK");
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void recognizeHand(
  Mat &in, Mat &depth, Mat &depthBackground, Mat &proj, Mat &out,
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
  Mat output;
  transformFrame(in, output, tfmedSize, proj);
  if (depth.empty()) depth = Mat::zeros(in.size(), CV_32F);
  transformFrame(depth, depth, tfmedSize, proj);
  if (depthBackground.empty()) depthBackground = Mat::zeros(in.size(), CV_32F);
  transformFrame(depthBackground, depthBackground, tfmedSize, proj);
  
  {
    std::ofstream depthStream;
    depthStream.open("depth.txt");
    for (int i = 0; i < depth.rows; i++)
    {
      for (int j = 0; j < depth.cols; j++)
      {
        depthStream << depth.at<float>(i,j) << ",";
      }
      depthStream << "\n";
    }
  }

  // imshow("debug", depth);
  // cv::waitKey(30);

  // step 2: find hand + finger
  // output.convertTo(output, CV_8UC3);
  vision::hand::processFrame(output, depth, depthBackground, output, handData, opts, debug);
  // imshow("2", output);

  // debugging...
  if (debug) {
    // std::cout << vision::hand::frameWithHandsToJSONString(handData) << std::endl;
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

  Mat proj = Mat::eye(3,3,CV_32F);
  Value projection = msg["data"]["projection"];
  if (projection.isArray() && projection.size() == 3*3) {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        proj.at<float>(i, j) = projection[(i*3)+j].asFloat();
  }

  Mat image, depthImage, depthBackground;
  uploadedOrCapturedImage(server, sender, msg, image, depthImage, depthBackground);

  {
    auto depthPercentile = msg["data"].get("depthPercentile", 0.0f).asFloat();
    auto s = cv::Size(maxWidth, maxHeight);
    transformFrame(depthImage, depthImage, s, proj);
    {
      cv::FileStorage fs("depth.yml", cv::FileStorage::WRITE);
      fs << "depth" << depthImage;
      fs.release();
    }

    convertToProperGrayscale(depthImage, depthPercentile);
    // imshow("depth", depthImage);
    transformFrame(depthBackground, depthBackground, s, proj);
    // {
    //   cv::FileStorage fs("depthBackground.yml", cv::FileStorage::WRITE);
    //   fs << "depth" << depthBackground;
    //   fs.release();
    // }
    convertToProperGrayscale(depthBackground, depthPercentile);
    // imshow("depthBg", depthBackground);
    // cv::waitKey(30);
  }

  {
    // Mat diff(depthImage.size(), CV_32F);
    // for (int i = 0; i < depthImage.rows; i++)
    // {
    //   for (int j = 1; j < depthImage.cols; j++)
    //   {
    //     diff.at<float>(i,j) = std::abs(depthImage.at<float>(i,j)-depthImage.at<float>(i,j-1));
    //   }
    // }
    Mat diff = depthBackground - depthImage;
    // absdiff(depthImage, depthBackground, diff);
    auto depthPercentile = msg["data"].get("depthPercentile", 0.0f).asFloat();
    convertToProperGrayscale(diff, depthPercentile);

    auto depthBlur = msg["data"].get("depthBlur", 0).asInt();
    if (depthBlur != 0)
      cv::blur(diff, diff, cv::Size(depthBlur, depthBlur));
    auto depthThreshold = msg["data"].get("depthThreshold", 0).asFloat();
    if (depthThreshold != 0.0f)
      cv::threshold(diff, diff, depthThreshold, 1, CV_THRESH_BINARY);

    cv::FileStorage fs("depth-diff.yml", cv::FileStorage::WRITE);
    fs << "depth" << diff;
    fs.release();

    diff *= 255;
    // cvtColor(diff,diff,cv::COLOR_GRAY2BGR);
    diff.convertTo(diff, CV_8UC3);
    cvdbg::recordImage(diff, "diff");
  }

  bool debug = msg["data"]["debug"].asBool();
  vision::hand::Options opts;
  handOptions(msg["data"], opts);
  vision::hand::FrameWithHands handData;
  Mat recorded;
  recognizeHand(image, depthImage, depthBackground, proj, recorded, handData, maxWidth, maxHeight, opts, debug);

  sendMat(recorded, server, sender);

  server->answer(msg, (string)"OK");
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

std::map<std::string, bool> handDetectionActivities;

void runHandDetectionProcessFor(
  string &target, Server &server,
  Mat &frame, Mat &depthFrame, Mat &depthBackground, Mat &proj, vision::cam::CameraPtr &dev,
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
    dev->readWithDepth(frame, depthFrame);

    vision::hand::FrameWithHands handData;
    Mat recorded;
    recognizeHand(frame, depthFrame, depthBackground, proj, recorded, handData, maxWidth, maxHeight, opts, debug);
    sendMat(recorded, server, target);
    // sendMat(frame, server, target);
    server->setTimer(10, bind(runHandDetectionProcessFor,
      target, server,
      frame, depthFrame, depthBackground, proj, dev,
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
  Mat frame, depthFrame, depthBackground;

  handDetectionActivities[sender] = true;

  Mat proj = Mat::eye(3,3,CV_32F);
  Value projection = msg["data"]["projection"];
  if (projection.size() == 3*3) {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        proj.at<float>(i, j) = projection[(i*3)+j].asFloat();
  }

  auto depthBackgroundFile = msg["data"].get("depthBackgroundFile", "").asString();
  if (depthBackgroundFile != "") {
    cv::FileStorage fs(depthBackgroundFile, cv::FileStorage::READ);
    fs["depth"] >> depthBackground;
  }

  vision::hand::Options opts;
  handOptions(msg["data"], opts);

  runHandDetectionProcessFor(sender, server, frame, depthFrame, depthBackground, proj, cam, maxWidth, maxHeight, opts, debug);

  server->answer(msg, (string)"OK");
}

}
}
