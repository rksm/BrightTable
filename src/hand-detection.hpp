#ifndef LIVE_TABLE_HAND_DETECTION_H_
#define LIVE_TABLE_HAND_DETECTION_H_

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "cv-helper.hpp"

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// hand detection

struct Finger {
  cv::Point base1;
  cv::Point base2;
  cv::Point tip;
  double angle() const { return angleBetween(base1, base2, tip); };
  bool operator==(const Finger &f) { return base1 == f.base1 && base2 == f.base2 && tip == f.tip; };
  friend std::ostream& operator<< (std::ostream& o, const Finger &f)
  {
    return o << "Finger<"
        << f.base1 << "," << f.base2 << "," << f.tip
        // << radToDeg(f.angle()) << ">" << std::endl;
        << std::endl;
  }
};

struct HandData {
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
FrameWithHands processFrame(cv::Mat, bool = false, cv::Mat = cv::Mat::eye(3, 3, CV_32F), cv::Size = cv::Size(400,300));

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
std::string frameWithHandsToJSONString(FrameWithHands data);

#endif  // LIVE_TABLE_HAND_DETECTION_H_
