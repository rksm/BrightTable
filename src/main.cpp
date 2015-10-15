#include "test-files.hpp"
#include "vision/hand-detection.hpp"
#include "vision/screen-detection.hpp"
#include "json/json.h"

using std::string;
using std::vector;
using std::regex;
using cv::Mat;
using cv::imread;

Mat transformFrame(Mat input, cv::Size tfmedSize, Mat &proj)
{
  input = resizeToFit(input, 700, 700);
  cv::Mat output = Mat(tfmedSize, CV_8UC3);
  cv::warpPerspective(input, output, proj, output.size());
  return output;
}

int main(int argc, char** argv)
{
  Mat frame;
  bool debug = true;
  int videoDevNo = 0;

  // cv::Size tfmedSize(432, 820);
  cv::Size tfmedSize(820, 432);

  // const string mode = "capture-one-video-frame";
  // const string mode = "recognize-test-files";
  // const string mode = "recognize-one-video-frame";
  const string mode = "transform-screen-and-recognize-file";
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
    // vector<string> testFiles{"/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/hand-test-white-2.png"};
    vector<string> testFiles = findTestFiles(regex("^hand-test-white-[0-9]+\\.[a-z]+$"));
    for (auto file : testFiles) {
        Mat result = extractLargestRectangle(resizeToFit(cv::imread(file, CV_LOAD_IMAGE_COLOR), 700, 700), tfmedSize, true);
        imwrite(regex_replace(file, regex("\\.[a-z]+$"), "-tfmed.png"), result);
        saveRecordedImages(regex_replace(file, regex("\\.[a-z]+$"), "-tfmed-debug.png"));
    }
  }
  else if (mode == "transform-screen-and-recognize-file")
  {
    vector<string> testFiles{"/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/hand-test-white-1.png"};
    // vector<string> testFiles = findTestFiles(regex("^hand-test-white-[0-9]+\\.[a-z]+$"));
    for (auto file : testFiles) {
      cv::Mat input = resizeToFit(cv::imread(file, CV_LOAD_IMAGE_COLOR), 700, 700);
      cv::Mat proj = screenProjection(input, tfmedSize, true);

      cv::Mat input2 = resizeToFit(cv::imread("/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/hand-test-white-3.png", CV_LOAD_IMAGE_COLOR), 700, 700);
      cv::Mat output = transformFrame(input2, tfmedSize, proj);
      processFrame(output, output, true);

      imwrite(regex_replace(file, regex("\\.[a-z]+$"), "-tfmed.png"), output);
      saveRecordedImages(regex_replace(file, regex("\\.[a-z]+$"), "-tfmed-debug.png"));
      std::cout << proj << std::endl;
    }
  }
  else if (mode == "recognize-one-video-frame")
  {
      cv::VideoCapture cap = cv::VideoCapture(videoDevNo);
      cap.read(frame);
      Mat out = Mat::zeros(frame.size(), frame.type());
      auto handData = processFrame(frame, out, true);
      std::cout << frameWithHandsToJSONString(handData) << std::endl;
  }
  else if (mode == "recognize-test-files")
  {
      // vector<string> testFiles{"/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/hand-test-white-1-tfmed.png"};
      vector<string> testFiles = findTestFiles(std::regex("hand-test-white-[0-9]+-tfmed.png"));
      for (auto file : testFiles) {
          Mat in = cv::imread(file, CV_LOAD_IMAGE_COLOR),
              out = Mat::zeros(frame.size(), frame.type());
          FrameWithHands handData;

          try {
            handData = processFrame(in, out, true);
          } catch(const std::exception& e) {
            saveRecordedImages(std::regex_replace(file, regex(".png$"), "-debug.png"));
          }

          saveRecordedImages(std::regex_replace(file, regex(".png$"), "-debug.png"));
          std::cout
            << frameWithHandsToJSONString(handData)
            << std::endl;
          cv::imwrite(std::regex_replace(file, regex(".png$"), "-result.png"), out);
      }
  }
  else if (mode == "convert-video") {
    cv::VideoCapture cap = cv::VideoCapture(videoDevNo);
    Mat frame;

    // 1. define screen area...
    for(;;) {
      cap >> frame; // get a new frame from camera
      imshow("out", resizeToFit(frame, 700,700));
      if (cv::waitKey(30) >= 0) break;
    }

    // show transformed area to check the projection...
    frame = resizeToFit(frame, 700, 700);
    Mat proj = screenProjection(frame, tfmedSize, true);
    Mat output = transformFrame(frame, tfmedSize, proj);
    // imshow("out", resizeToFit(getAndClearRecordedImages(), 820, 820));
    imshow("out", output);
    for(;;) if (cv::waitKey(30) >= 0) break;

    for(;;)
    {
      cap >> frame;
      Mat output = transformFrame(resizeToFit(frame, 700, 700), tfmedSize, proj);
      processFrame(output, output, true);
      imshow("out", resizeToFit(getAndClearRecordedImages(), 820, 820));
      if (cv::waitKey(30) >= 0) break;
    }
  }

  return 0;

}
