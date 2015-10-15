#include "vision/quad-transform.hpp"
#include <vision/cv-helper.hpp>
#include <algorithm>

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Corners
std::vector<cv::Point2f> Corners::asVector() {
  return std::vector<cv::Point2f> {
    topLeft,topRight,bottomRight,bottomLeft};
};

bool Corners::empty() const { Corners empty{}; return &empty == this; };

bool Corners::operator==(const Corners &other) const {
  return topLeft == other.topLeft
      && topRight == other.topRight
      && bottomRight == other.bottomRight
      && bottomLeft == other.bottomLeft;
};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

using cv::Point2f;
using cv::Vec4i;
using cv::Mat;
using cv::Rect;
using std::vector;

Point2f computeIntersect(Vec4i a, Vec4i b)
{
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

    if (float d = ((float)(x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4)))
    {
        Point2f pt;
        pt.x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d;
        pt.y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d;
        return pt;
    }
    else
        return Point2f(-1, -1);
}

// // http://opencv-code.com/tutorials/automatic-perspective-correction-for-quadrilateral-objects/
// std::vector<Point2f> findCorners(vector<Vec4i> lines)
// {
//   std::vector<Point2f> corners;
//   for (int i = 0; i < lines.size(); i++)
//   {
//       for (int j = i+1; j < lines.size(); j++)
//       {
//           Point2f pt = computeIntersect(lines[i], lines[j]);
//           if (pt.x >= 0 && pt.y >= 0)
//               corners.push_back(pt);
//       }
//   }
//   return corners;
// }

const int slopeEps = 3;

bool similarSlope(Vec4i l1, Vec4i l2) {
  auto slope1 = (double)(l1[3] - l1[1]) / (double)(l1[2] - l1[0]),
       slope2 = (double)(l2[3] - l2[1]) / (double)(l2[2] - l1[0]);

  // std::cout << slope1 << "..." << slope2 << std::endl;
  return (std::max(slope1, slope2) - std::min(slope1, slope2)) < slopeEps;
}

std::vector<Point2f> findCorners(vector<Vec4i> lines)
{
  // returns cloud of points. We later analyze it to find the 4 real corners
  std::vector<Point2f> corners;
  for (int i = 0; i < lines.size(); i++)
  {
    for (int j = i+1; j < lines.size(); j++)
    {
      Point2f pt = computeIntersect(lines[i], lines[j]);
      if (pt.x >= 0 && pt.y >= 0) {
          Point2f p1(lines[i][0],lines[i][1]),
                  p2(lines[j][0],lines[j][1]);
          float angle = radToDeg(angleBetween(p1, p2, pt));
          // std::cout << angle << std::endl;
          if ((angle > 70) && (angle < 130)) corners.push_back(pt);
      }
    }
  }
  return corners;
}

bool isQuadrilateral(vector<Point2f> corners)
{
  vector<Point2f> approx;
  cv::approxPolyDP(Mat(corners), approx,
     cv::arcLength(Mat(corners), true) * 0.02, true);
    // std::cout << approx.size() << std::endl;
  return approx.size() == 4;
}

Corners sortBoundingBoxPoints(vector<Point2f> &points, Point2f center, Rect fromRect)
{
    // returns top-left, top-right, bottom-right, and bottom-left.
    std::vector<Point2f> tls, bls, trs, brs;

    for (auto p : points)
    {
      if (p.y < center.y)
        if (p.x < center.x) tls.push_back(p);
        else trs.push_back(p);
      else
        if (p.x < center.x) bls.push_back(p);
        else brs.push_back(p);
    }

    if (tls.empty() || bls.empty() || trs.empty() || brs.empty()) return Corners{};

    auto tl_it = min_element(tls.begin(), tls.end(), [&](Point2f p1, Point2f p2) { auto ref = Point2f(fromRect.x,     fromRect.y);      return norm(ref - p1) < norm(ref - p2); }),
         tr_it = min_element(trs.begin(), trs.end(), [&](Point2f p1, Point2f p2) { auto ref = Point2f(fromRect.width, fromRect.y);      return norm(ref - p1) < norm(ref - p2); }),
         bl_it = min_element(bls.begin(), bls.end(), [&](Point2f p1, Point2f p2) { auto ref = Point2f(fromRect.x,     fromRect.height); return norm(ref - p1) < norm(ref - p2); }),
         br_it = min_element(brs.begin(), brs.end(), [&](Point2f p1, Point2f p2) { auto ref = Point2f(fromRect.width, fromRect.height); return norm(ref - p1) < norm(ref - p2); });

    return Corners{*tl_it,*tr_it,*br_it,*bl_it};
}

Corners pointsSorted(vector<Point2f> &boundingBoxPoints, Rect fromRect)
{
  // Get mass center
  Point2f center(0,0);
  for (int i = 0; i < boundingBoxPoints.size(); i++)
    center += boundingBoxPoints[i];

  center *= (1. / boundingBoxPoints.size());

  return sortBoundingBoxPoints(boundingBoxPoints, center, fromRect);
}

Mat boundingBoxPointsToRectTransform(vector<Point2f> &boundingBoxPoints, Rect fromRect, Rect destBounds)
{
  // Corners = four points defining a quadrilateral; width, height defining a
  // target rectangle

  Corners corners = pointsSorted(boundingBoxPoints, fromRect);
  if (corners.empty()) return Mat::eye(3,3,CV_32F);

  // std::cout << corners << std::endl;
  std::vector<Point2f> quadPoints {
    Point2f(destBounds.x, destBounds.y),
    Point2f(destBounds.width, destBounds.y),
    Point2f(destBounds.width, destBounds.height),
    Point2f(destBounds.x, destBounds.height)
  };

  // Get transformation matrix
  // std::rotate(corners.rbegin(), corners.rbegin() + 1, corners.rend());
  return cv::getPerspectiveTransform(corners.asVector(), quadPoints);
}

Mat projectLinesTransform(vector<Vec4i> lines, Rect fromRect, Rect intoRect)
{
  // takes a set of (quadrilateral) lines and returns a perspective transform
  // to map stuff into the area defined by intoRect
  vector<Point2f> corners = findCorners(lines);
  corners.erase(std::remove_if(corners.begin(), corners.end(),
    [&](Point2f p) { return !fromRect.contains(p); }), corners.end());

  // Mat area = Mat::zeros(fromRect.size(), CV_8UC1);
  // for (auto p : corners) {
  //   circle(area, p, 10, cv::Scalar(255,0,0), 3);
  // }
  // recordImage(area, "??????");

  // assert(all_of(corners.begin(), corners.end(),
  //   [&](cv::Point2f p){ return fromRect.contains(p); }));

  return boundingBoxPointsToRectTransform(corners, fromRect,intoRect);
}
