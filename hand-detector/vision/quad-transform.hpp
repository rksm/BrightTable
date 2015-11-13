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
  Corners() : Corners(0,0,0,0,0,0,0,0) {};
  Corners(cv::Point2f tl,cv::Point2f tr,cv::Point2f br,cv::Point2f bl) :
    topLeft(tl), topRight(tr), bottomRight(br), bottomLeft(bl) {};
  Corners(
    float tlX, float tlY,
    float trX, float trY,
    float brX, float brY,
    float blX, float blY) :
      topLeft(cv::Point2f(tlX, tlY)),
      topRight(cv::Point2f(trX, trY)),
      bottomRight(cv::Point2f(brX, brY)),
      bottomLeft(cv::Point2f(blX, blY)) {};
  cv::Point2f topLeft;
  cv::Point2f topRight;
  cv::Point2f bottomRight;
  cv::Point2f bottomLeft;
  std::vector<cv::Point2f> asVector();
  bool empty() const;
  bool operator==(const Corners &other) const;
};

cv::Mat cornerTransform(Corners &corners,  cv::Rect&);
Corners findCorners(std::vector<cv::Vec4i>&, cv::Rect&);

#endif  // QUAD_TRANSFORM_H_INCLUDE
