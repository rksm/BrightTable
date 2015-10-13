#include "HandDetection.h"
#include <regex>

#include "json/json/json.h"

using cv::Mat;
using cv::imread;

int main(int argc, char** argv)
{
    Mat frame;
    bool debug = true;
    int videoDevNo = 1;

    // const string mode = "convert-video";
    // const string mode = "capture-one-video-frame";
    // const string mode = "recognize-test-files";
    // const string mode = "recognize-one-video-frame";
    const string mode = "convert-video";

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
    else if (mode == "recognize-one-video-frame")
    {
        cv::VideoCapture cap = cv::VideoCapture(videoDevNo);
        cap.read(frame);
        std::cout << frameWithHandsToJSONString(processFrame(frame, true)) << std::endl;
    }
    else if (mode == "recognize-test-files")
    {
        std::regex reg("\\.[a-z]+$");
        // vector<string> testFiles{"/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/hand-test-13.png"};
        vector<string> testFiles = findTestFiles();
        for (auto file : testFiles) {
            // std::cout
            //   << frameWithHandsToJSONString(processFrame(cv::imread(file, CV_LOAD_IMAGE_COLOR), true))
            //   << std::endl;
            processFrame(cv::imread(file, CV_LOAD_IMAGE_COLOR), true);
            saveRecordedImages(std::regex_replace(file, reg, "-result.png"));
        }
    }
    else if (mode == "convert-video") {
      cv::VideoCapture cap = cv::VideoCapture(videoDevNo);
      for(;;)
      {
          cap >> frame; // get a new frame from camera
          // std::cout << frameWithHandsToJSONString(processFrame(frame, true)) << std::endl;
          processFrame(frame, true);
          imshow("out", resizeToFit(getAndClearRecordedImages(), 1400, 500));
          if (cv::waitKey(30) >= 0) break;
      }
    }

    return 0;

}
