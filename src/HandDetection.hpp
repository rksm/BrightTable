#ifndef HANDDETECTION_H_
#define HANDDETECTION_H_

#include <stdio.h>
#include <opencv2/opencv.hpp>

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// helpers
inline double angleBetween(cv::Point2d p1, cv::Point2d p2, cv::Point2d c = cv::Point(0.0,0.0))
{
  cv::Point2d q1 = p1 - c, q2 = p2 - c;
  return acos((q1 * (1/cv::norm(q1))).ddot(q2 * (1/cv::norm(q2))));
}

inline double radToDeg(double r) { return 180/CV_PI * r; };

template<typename PointType>
void drawPointsConnected(std::vector<PointType> points, cv::Mat &out)
{
  std::vector<PointType> pointsCopy(points.size());
  std::rotate_copy(points.begin(), points.begin()+1, points.end(), pointsCopy.begin());
  for (auto it = points.begin(), it2 = pointsCopy.begin(); it != points.end(); it++, it2++) {
    line(out, *it, *it2, cv::Scalar(255,255,0), 2);
  }
}
template void drawPointsConnected<cv::Point2f >(std::vector<cv::Point2f >, cv::Mat&);
template void drawPointsConnected<cv::Point >(std::vector<cv::Point >, cv::Mat&);

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
FrameWithHands processFrame(cv::Mat, bool = false);

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// screen detection
cv::Mat extractLargestRectangle(cv::Mat, cv::Size, bool = false);

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
cv::Mat resizeToFit(cv::Mat, int, int);
const cv::Scalar randomColor();

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
void recordImage(const cv::Mat, std::string);
void saveRecordedImages(const std::string&);
cv::Mat getAndClearRecordedImages();

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
std::vector<std::string> findTestFiles();

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
std::string frameWithHandsToJSONString(FrameWithHands data);

#endif  // HANDDETECTION_H_
