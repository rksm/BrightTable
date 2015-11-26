#ifndef LIVE_TABLE_HAND_DETECTION_H_
#define LIVE_TABLE_HAND_DETECTION_H_

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "vision/cv-helper.hpp"
#include "json/forwards.h"

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// hand detection

namespace vision {
namespace hand {

struct Options
{
  // How far apart can convexity defect hull points lay apart to still be
  // considered as one fingertip?
  bool debug = false;
  bool renderDebugImages = true;
  int fingerTipWidth = 50;
  float minHandAreaInPercent = 2.0f;
  int depthSamplingKernelLength = 10;
  int blurIntensity = 11;
  int thresholdMin = 100;
  int thresholdMax = 245;
  int thresholdType = CV_THRESH_BINARY_INV;
  int dilateIterations = 5;
  int cropWidth = 12;
};

struct HandContour
{
  cv::RotatedRect bounds;
  std::vector<cv::Point> contourPoints;
  cv::Point armStart;
  cv::Point pointTowards;
  int fingerRadius;
  cv::Point palmCenter;
};

struct Finger
{
  cv::Point base1;
  cv::Point base2;
  cv::Point tip;
  int z;
  int length() const { return std::max(norm(base1-tip), norm(base2-tip)); };
  double angle() const { return cvhelper::angleBetween(base1, base2, tip); };
  bool operator==(const Finger &f) { return base1 == f.base1 && base2 == f.base2 && tip == f.tip && z == f.z; };
  friend std::ostream& operator<< (std::ostream& o, const Finger &f)
  {
    return o << "Finger<"
        << f.base1 << "," << f.base2 << "," << f.tip << "," << f.z
        // << cvhelper::radToDeg(f.angle()) << ">" << std::endl;
        << std::endl;
  }
};

struct HandData
{
  int palmRadius; // min(width, height) / 2 if contourBounds
  cv::Point palmCenter; // min(width, height) / 2 if contourBounds
  cv::RotatedRect contourBounds;
  cv::RotatedRect convexityDefectArea; // the subset of the contourBounds that we consider for defects
  std::vector<Finger> fingerTips;
};

struct FrameWithHands {
  std::time_t time;
  cv::Size imageSize;
  std::vector<HandData> hands;
};

void processFrame(cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, FrameWithHands&, Options);

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
Json::Value frameWithHandsToJSON(FrameWithHands &data);
std::string frameWithHandsToJSONString(FrameWithHands &data);

}
}

#endif  // LIVE_TABLE_HAND_DETECTION_H_
