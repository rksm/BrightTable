#ifndef CV_HELPER_H_
#define CV_HELPER_H_

#include <opencv2/opencv.hpp>

namespace cvhelper
{

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// helpers
template<typename NumType>
double angleBetween(cv::Point_<NumType> p1, cv::Point_<NumType> p2, cv::Point_<NumType> c = cv::Point_<NumType>(0,0))
{
  cv::Point_<NumType> q1 = p1 - c, q2 = p2 - c;
  return acos((q1 * (1/cv::norm(q1))).ddot(q2 * (1/cv::norm(q2))));
}
template double angleBetween<double>(cv::Point_<double> p1, cv::Point_<double> p2, cv::Point_<double> c);
template double angleBetween<float>(cv::Point_<float> p1, cv::Point_<float> p2, cv::Point_<float> c);
template double angleBetween<int>(cv::Point_<int> p1, cv::Point_<int> p2, cv::Point_<int> c);

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
void resizeToFit(cv::Mat&, cv::Mat&, float, float);

const cv::Scalar randomColor();

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void convertToProperGrayscale(cv::Mat&, float percentile=5.0f);

}

#endif  // CV_HELPER_H_