#include "test-files.hpp"
#include "vision/hand-detection.hpp"
#include "vision/screen-detection.hpp"
#include "json/json.h"
#include "timer.hpp"

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
  // const string mode = "transform-screen-and-recognize-file";
  // const string mode = "transform-screen-file";
  const string mode = "convert-video";

  if (mode == "capture-one-video-frame")
  {
      cv::VideoCapture cap = cv::VideoCapture(videoDevNo);
      cap.read(frame);
      cv::imwrite("/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/other-webcam-issue-2.png", frame);
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
    vector<string> testFiles{"/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/other-webcam.png"};
    // vector<string> testFiles = findTestFiles(regex("^hand-test-white-[0-9]+\\.[a-z]+$"));
    for (auto file : testFiles) {
      Mat input = resizeToFit(cv::imread(file, CV_LOAD_IMAGE_COLOR), 700, 700); 
      Mat result = extractLargestRectangle(input, tfmedSize, true);
      cv::Mat proj = screenProjection(input, tfmedSize, true);
      std::cout << proj << std::endl;
      imwrite(regex_replace(file, regex("\\.[a-z]+$"), "-tfmed.png"), result);
      saveRecordedImages(regex_replace(file, regex("\\.[a-z]+$"), "-tfmed-debug.png"));
    }
  }
  else if (mode == "transform-screen-and-recognize-file")
  {
    vector<string> testFiles{"/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/other-webcam-issue-2.png"};
    // vector<string> testFiles = findTestFiles(regex("^hand-test-white-[0-9]+\\.[a-z]+$"));
    for (auto file : testFiles) {
      cv::Mat input = resizeToFit(cv::imread(file, CV_LOAD_IMAGE_COLOR), 700, 700);
      // cv::Mat proj = screenProjection(input, tfmedSize, true);

      // cv::Mat proj = cv::Mat::zeros(3,3, CV_32FC1);
      // proj.at<float>(0,0) = 1.311824869288347;
      // proj.at<float>(0,1) = 0.5819373495118125;
      // proj.at<float>(0,2) = -245.2718043772442;
      // proj.at<float>(1,0) = -0.2655269490085852;
      // proj.at<float>(1,1) = 1.286784500521986;
      // proj.at<float>(1,2) = 19.15879094694175;;
      // proj.at<float>(2,0) = -0.0002367759237758272;
      // proj.at<float>(2,1) = 0.0005972594927079796;
      // proj.at<float>(2,2) = 1;

      cv::Mat proj = cv::Mat::zeros(3,3, CV_32FC1);
      proj.at<float>(0,0) = 1.652389079784966;
      proj.at<float>(0,1) = 0.5326780287383337;
      proj.at<float>(0,2) = -222.7898690137106;;
      proj.at<float>(1,0) = -0.0363833634945249;
      proj.at<float>(1,1) = 1.649966529588601;
      proj.at<float>(1,2) = -15.15386596733211;
      proj.at<float>(2,0) = -8.42207531049395e-05;
      proj.at<float>(2,1) = 0.001051777679467157;
      proj.at<float>(2,2) = 1;


      cv::Mat output = transformFrame(input, tfmedSize, proj);
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
      vector<string> testFiles{"/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/hand-test-white-4-tfmed.png"};
      // vector<string> testFiles = findTestFiles(std::regex("hand-test-white-[0-9]+-tfmed.png"));
      for (auto file : testFiles) {
          Mat in = cv::imread(file, CV_LOAD_IMAGE_COLOR),
              out = Mat::zeros(frame.size(), frame.type());
          FrameWithHands handData = processFrame(in, out, true);
          saveRecordedImages(std::regex_replace(file, regex(".png$"), "-debug.png"));
          std::cout
            << frameWithHandsToJSONString(handData)
            << std::endl;
          cv::imwrite(std::regex_replace(file, regex(".png$"), "-result.png"), out);
      }
  }
  else if (mode == "convert-video") {
    cv::VideoCapture cap = cv::VideoCapture(videoDevNo);
    bool debug = true;
    Mat frame;

    // 1. define screen area...
    for(;;) {
      cap >> frame;
      imshow("out", resizeToFit(frame, 700,700));
      if (cv::waitKey(30) >= 0) break;
    }

    Mat proj;
    for(;;) {
      cap >> frame;
      // show transformed area to check the projection...
      frame = resizeToFit(frame, 700, 700);
      proj = screenProjection(frame, tfmedSize, debug);
      if ((proj.at<int>(0,0) != 0) && (proj.at<int>(0,1) != 0)) break;
      if (cv::waitKey(30) >= 0) break;
    }

    for(;;) {
      // std::cout << proj << std::endl;
      Mat output = transformFrame(frame, tfmedSize, proj);
      // imshow("out", resizeToFit(getAndClearRecordedImages(), 820, 820));
      imshow("out", resizeToFit(output, 700,700));
      if (cv::waitKey(30) >= 0) break;
    }

    std::chrono::system_clock clock;
    auto t = clock.now();
    while(1)
    {
      cap.read(frame);
      // std::cout << timeToRunMs([&](){
        Mat output = transformFrame(resizeToFit(frame, 700, 700), tfmedSize, proj);
        auto handData = processFrame(output, output, debug);
        std::cout << frameWithHandsToJSONString(handData) << std::endl;
      // }).count() << std::endl;
      imshow("out", resizeToFit(getAndClearRecordedImages(), 1400,1400));
      if (cv::waitKey(10) >= 0) break;

      // auto now = clock.now();
      // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - t);
      // t = now;
      // std::cout << duration.count() << std::endl;

    }
  }

  return 0;

}
