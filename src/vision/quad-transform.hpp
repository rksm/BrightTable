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

std::vector<cv::Point2f> findCorners(std::vector<cv::Vec4i>);
cv::Mat cornersToRectTransform(std::vector<cv::Point2f>&, cv::Rect);
cv::Mat projectLinesTransform(std::vector<cv::Vec4i>&,  cv::Rect, cv::Rect);
void projectLinesInto(std::vector<cv::Vec4i>&, cv::Mat&, cv::Mat&);

#endif  // QUAD_TRANSFORM_H_INCLUDE
