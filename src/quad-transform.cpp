#include "quad-transform.h"
#include <algorithm>

using cv::Point2f;
using cv::Vec4i;
using cv::Mat;
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

// http://opencv-code.com/tutorials/automatic-perspective-correction-for-quadrilateral-objects/
std::vector<Point2f> findCorners(vector<Vec4i> lines)
{
  std::vector<Point2f> corners;
  for (int i = 0; i < lines.size(); i++)
  {
      for (int j = i+1; j < lines.size(); j++)
      {
          Point2f pt = computeIntersect(lines[i], lines[j]);
          if (pt.x >= 0 && pt.y >= 0)
              corners.push_back(pt);
      }
  }
  return corners;
}

bool isQuadrilateral(vector<Point2f> corners)
{
  vector<Point2f> approx;
  cv::approxPolyDP(Mat(corners), approx,
     cv::arcLength(Mat(corners), true) * 0.02, true);
    std::cout << approx.size() << std::endl;
  return approx.size() == 4;
}

void sortCorners(vector<Point2f>& corners, Point2f center)
{
    // returns top-left, top-right, bottom-right, and bottom-left.
    std::vector<Point2f> top, bot;

    for (auto p : corners)
    {
      if (p.y < center.y) top.push_back(p);
      else bot.push_back(p);
    }

    Point2f tl = top[0].x > top[1].x ? top[1] : top[0],
            tr = top[0].x > top[1].x ? top[0] : top[1],
            bl = bot[0].x > bot[1].x ? bot[1] : bot[0],
            br = bot[0].x > bot[1].x ? bot[0] : bot[1];

    corners.clear();
    corners.push_back(tl);
    corners.push_back(tr);
    corners.push_back(br);
    corners.push_back(bl);
}

void cornersSorted(vector<Point2f> &corners)
{
  // Get mass center
  Point2f center(0,0);
  for (int i = 0; i < corners.size(); i++)
    center += corners[i];

  center *= (1. / corners.size());
  sortCorners(corners, center);
}

Mat cornersToRectTransform(vector<Point2f> &corners, cv::Rect destBounds)
{
  // Corners = four points defining a quadrilateral; width, height defining a
  // target rectangle

  cornersSorted(corners);  

  std::vector<Point2f> quadPoints {
    Point2f(destBounds.x, destBounds.y),
    Point2f(destBounds.width, destBounds.y),
    Point2f(destBounds.width, destBounds.height),
    Point2f(destBounds.x, destBounds.height)
  };

  // Get transformation matrix
  return cv::getPerspectiveTransform(corners, quadPoints);
}

Mat projectLinesTransform(vector<Vec4i> &lines, Mat &fromMat, cv::Rect intoRect)
{
  // takes a set of (quadrilateral) lines and returns a perspective transform
  // to map stuff into the area defined by intoRect

  vector<Point2f> corners = findCorners(lines);
  std::remove_if(corners.begin(), corners.end(), [&](Point2f p) {
    return (p.x < 0) || (p.y < 0)
        || (p.x > fromMat.cols) || (p.y > fromMat.rows);
  });

  return cornersToRectTransform(corners, intoRect);
}

void projectLinesInto(vector<Vec4i> &lines, Mat &fromMat, Mat &intoMat)
{
  cv::Rect projectionRect(cv::Point(0,0), intoMat.size());
  Mat tfm = projectLinesTransform(lines, fromMat, projectionRect);
  cv::warpPerspective(fromMat, intoMat, tfm, projectionRect.size());
}
