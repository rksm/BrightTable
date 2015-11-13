#include "vision/screen-detection.hpp"
#include "vision/quad-transform.hpp"
#include "vision/cv-helper.hpp"

#include <numeric>
#include <algorithm>

using std::vector;
using cv::Mat;
using cv::Point;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void prepareImage(const Mat &imgIn, Mat &imgOut)
{
  if (imgIn.type() == CV_8UC3) cvtColor(imgIn, imgOut, CV_BGR2GRAY);
  else imgIn.copyTo(imgOut);
  medianBlur(imgOut, imgOut, 21);
  threshold(imgOut, imgOut, 100, 245, CV_THRESH_BINARY);
  // cvtColor(imgOut, imgOut, CV_RGB2GRAY);
}

Mat extractContours(const Mat &imgIn, bool debugRecordings)
{
  
  Mat dst = Mat::zeros(imgIn.size(), CV_8UC1);
  Mat contourDst = Mat::zeros(imgIn.size(), CV_8UC1);
  prepareImage(imgIn, dst);
  if (debugRecordings) cvhelper::recordImage(dst, "prep");

  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  findContours(dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));

  double area = 0;
  int biggestI = 0;
  for (int i = 0; i < contours.size(); i++) {
    auto areaNow = contourArea(contours[i]);
    if (areaNow > area) {
      biggestI = i;
      area = areaNow;
    }
  }

  vector<Point> hullPoints;
  convexHull(contours[biggestI], hullPoints);
  cvhelper::drawPointsConnected<Point>(hullPoints, contourDst);
  if (debugRecordings) cvhelper::recordImage(contourDst, "contour");

  return contourDst;
}

vector<cv::Vec4i> detectLines(Mat &imgIn)
{
  cv::Size size = imgIn.size();
  if (false)
  {
    vector<cv::Vec2f> lines;
    int minLineLength = std::min(size.width, size.height)*.5;
    HoughLines(imgIn, lines, 1, CV_PI/180, 150, 0, 0);

    vector<cv::Vec4i> outLines;
    for( size_t i = 0; i < lines.size(); i++ )
    {
      float rho = lines[i][0], theta = lines[i][1];
      Point pt1, pt2;
      double a = cos(theta), b = sin(theta);
      double x0 = a*rho, y0 = b*rho;
      pt1.x = cvRound(x0 + 1000*(-b));
      pt1.y = cvRound(y0 + 1000*(a));
      pt2.x = cvRound(x0 - 1000*(-b));
      pt2.y = cvRound(y0 - 1000*(a));
      outLines.push_back(cv::Vec4i {pt1.x, pt1.y,pt2.x, pt2.y});
    }
    return outLines;
  }
  else
  {
    int minLineLength = std::min(size.width, size.height)*.2;
    vector<cv::Vec4i> lines;
    // cv::HoughLinesP(imgIn, lines, 1, CV_PI/180, 40, minLineLength, 40);
    cv::HoughLinesP(imgIn, lines, 1, CV_PI/180, 170, 0, 0);
    lines.erase(remove_if(lines.begin(), lines.end(),
      [&](cv::Vec4i l){
        Point p1(l[0], l[1]), p2(l[2], l[3]);
        return cv::norm(p2-p1) < minLineLength;
      }), lines.end());
    return lines;
  }
}

vector<cv::Vec4i> findLinesOfLargestRect(Mat &src, bool debugRecordings)
{
  Mat contours = extractContours(src, debugRecordings);
  vector<cv::Vec4i> lines = detectLines(contours);
  
  if (debugRecordings)
  {
    for (auto l : lines) {
      line(src, Point(l[0], l[1]), Point(l[2], l[3]), cvhelper::randomColor(), 3, CV_AA);
    }
    cvhelper::recordImage(src, "lines");
  }
  return lines;
}

Mat screenProjection(Mat &src, cv::Size size, bool debugRecordings)
{
  auto lines = findLinesOfLargestRect(src, debugRecordings);
  auto from = cv::Rect(cv::Point(0,0), src.size());
  auto into = cv::Rect(cv::Point(0,0), size);
  auto corners = findCorners(lines, from);
  auto tfm = cornerTransform(corners, into);
  return tfm;
}

Mat applyScreenProjection(cv::Mat &in, cv::Mat &projection, cv::Size size, bool debugRecordings)
{
  cv::Mat projected = cv::Mat::zeros(size.width, size.height, CV_8UC3);
  cv::warpPerspective(in, projected, projection, size);
  if (debugRecordings) cvhelper::recordImage(projected, "projection");
  return projected;
}

Mat extractLargestRectangle(Mat &src, cv::Size size, bool debugRecordings)
{
  Mat proj = screenProjection(src, size, debugRecordings);
  return applyScreenProjection(src, proj, size, debugRecordings);
}

Corners cornersOfLargestRect(Mat &src, bool debugRecordings)
{
  auto lines = findLinesOfLargestRect(src, debugRecordings);
  auto bounds = cv::Rect(cv::Point(0,0), src.size());
  return findCorners(lines, bounds);
}
