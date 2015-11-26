#ifndef SCREEN_DETECTION_H_
#define SCREEN_DETECTION_H_

#include <opencv2/opencv.hpp>
#include <vision/quad-transform.hpp>

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// screen detection

namespace vision {
namespace screen {

struct Options
{
  bool debug = false;
  int blurIntensity = 21;
  int minThreshold = 100;
  int maxThreshold = 245;
  int thresholdType = CV_THRESH_BINARY;
  int houghRho = 1;
  float houghTheta = CV_PI / 180;
  int houghMinLineLength = 0;
  int houghMinLineGap = 0;
  quad::Options quadOptions;
};

cv::Mat extractLargestRectangle(cv::Mat&, cv::Size, Options, bool = false);
cv::Mat screenProjection(cv::Mat&, cv::Size, Options, bool = false);
quad::Corners cornersOfLargestRect(cv::Mat&, Options, bool = false);
cv::Mat applyScreenProjection(cv::Mat&, cv::Mat&, cv::Size, Options, bool = false);

}
}

#endif  // SCREEN_DETECTION_H_