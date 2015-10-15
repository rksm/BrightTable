#ifndef QUAD_TRANSFORM_H_INCLUDE
#define QUAD_TRANSFORM_H_INCLUDE

/*
see http://opencv-code.com/tutorials/automatic-perspective-correction-for-quadrilateral-objects/

For a given set of cv::Vec4i lines that can e.g. be optained using
cv::HoughLines, projectLinesLinesInto(lines, fromMat, toMat) will take the
(quadrilateral) area defined by lines and create a projection of this are from
fromMat into toMat
*/

#include <opencv2/opencv.hpp>

struct Corners
{
  cv::Point2f topLeft;
  cv::Point2f topRight;
  cv::Point2f bottomRight;
  cv::Point2f bottomLeft;
  std::vector<cv::Point2f> asVector();
  bool empty() const;
  bool operator==(const Corners &other) const;
};

cv::Mat projectLinesTransform(std::vector<cv::Vec4i>,  cv::Rect, cv::Rect);

#endif  // QUAD_TRANSFORM_H_INCLUDE
