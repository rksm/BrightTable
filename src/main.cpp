#include "hand-detection.hpp"
#include "test-files.hpp"
#include "screen-detection.hpp"
#include "json/json.h"

using std::string;
using std::vector;
using cv::Mat;
using cv::imread;

int main(int argc, char** argv)
{
  Mat frame;
  bool debug = true;
  int videoDevNo = 0;

  // const string mode = "capture-one-video-frame";
  const string mode = "recognize-test-files";
  // const string mode = "recognize-one-video-frame";
  // const string mode = "recognize-screen-file";ˇ
  // const string mode = "transform-screen-file";
  // const string mode = "convert-video";

  if (mode == "capture-one-video-frame")
  {
      cv::VideoCapture cap = cv::VideoCapture(videoDevNo);
      cap.read(frame);
      cv::imwrite("/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/foo.png", frame);
      for (;;) {
        imshow("debug", frame);
        if (cv::waitKey(30) >= 0) break;
      }
  }
  else if (mode == "recognize-screen-file")
  {
      Mat img = cv::imread("/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/sreen-raw.png", CV_LOAD_IMAGE_COLOR);
      Mat result = extractLargestRectangle(resizeToFit(img, 700, 700), cv::Size(1000,1000), true);
      saveRecordedImages("/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/screen-debug.png");
      cv::imwrite("/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/screen-recognized.png", result);
  }
  else if (mode == "recognize-screen-video")
  {
      cv::VideoCapture cap = cv::VideoCapture(videoDevNo);
      for (;;)
      {
        cap >> frame;
        Mat result = extractLargestRectangle(resizeToFit(frame, 500, 500), cv::Size(1000,1000), true);
        // imshow("out", resizeToFit(getAndClearRecordedImages(), 1400, 500));
        imshow("out", result);
        if (cv::waitKey(30) >= 0) break;
      }
  }
  else if (mode == "transform-screen-file")
  {
    cv::Size tfmedSize(432, 820);
    std::regex reg("\\.[a-z]+$");
    // vector<string> testFiles{"/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/hand-test-white-1.png"};
    vector<string> testFiles = findTestFiles(std::regex("^hand-test-white-[0-9]+\\.[a-z]+$"));
    for (auto file : testFiles) {
        Mat result = extractLargestRectangle(resizeToFit(cv::imread(file, CV_LOAD_IMAGE_COLOR), 700, 700), tfmedSize, true);
        imwrite(std::regex_replace(file, reg, "-result.png"), result);
        saveRecordedImages(std::regex_replace(file, reg, "-debug.png"));
    }
  }
  else if (mode == "recognize-one-video-frame")
  {
      cv::VideoCapture cap = cv::VideoCapture(videoDevNo);
      cap.read(frame);
      std::cout << frameWithHandsToJSONString(processFrame(frame, true)) << std::endl;
  }
  else if (mode == "recognize-test-files")
  {
      std::regex reg("\\.[a-z]+$");
      vector<string> testFiles{"/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/hand-test-white-tfmed-1.png"};
      // vector<string> testFiles = findTestFiles();
      for (auto file : testFiles) {
          // std::cout
          //   << frameWithHandsToJSONString(processFrame(cv::imread(file, CV_LOAD_IMAGE_COLOR), true))
          //   << std::endl;
          processFrame(imread(file, CV_LOAD_IMAGE_COLOR), true);
          saveRecordedImages(std::regex_replace(file, reg, "-debug.png"));
          // cv::imwrite(std::regex_replace(file, reg, "-result.png"));
      }
  }
  else if (mode == "convert-video") {
    cv::VideoCapture cap = cv::VideoCapture(videoDevNo);
    // cv::Size tfmedSize(432, 730);
    cv::Size tfmedSize(432, 820);
    // 1. define screen area...
    cap >> frame;
    imshow("out", resizeToFit(frame, 700,700));
    for(;;) if (cv::waitKey(30) >= 0) break;
    cap >> frame;
    Mat result = extractLargestRectangle(resizeToFit(frame, 820, 820), tfmedSize, true);
    imshow("out", resizeToFit(getAndClearRecordedImages(), 820, 820));
    cap >> frame;
    cv::Mat projTransform = screenProjection(resizeToFit(frame, 820, 820), tfmedSize, false);
    for(;;) if (cv::waitKey(30) >= 0) break;
    for(;;)
    {
        cap >> frame; // get a new frame from camera
        // std::cout << frameWithHandsToJSONString(processFrame(frame, true)) << std::endl;
        processFrame(frame, true, projTransform, tfmedSize);
        imshow("out", resizeToFit(getAndClearRecordedImages(), 820, 820));
        if (cv::waitKey(30) >= 0) break;
    }
  }

  return 0;

}
