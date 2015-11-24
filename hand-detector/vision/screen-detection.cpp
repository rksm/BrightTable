#include "vision/screen-detection.hpp"
#include "vision/quad-transform.hpp"
#include "vision/cv-helper.hpp"
#include "vision/cv-debugging.hpp"

#include <numeric>
#include <algorithm>


using std::vector;
using cv::Mat;
using cv::Point;

namespace vision {
namespace screen {

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void prepareImage(const Mat &imgIn, Mat &imgOut, Options opts)
{
  if (imgIn.type() == CV_8UC3) cvtColor(imgIn, imgOut, CV_BGR2GRAY);
  else imgIn.copyTo(imgOut);
  medianBlur(imgOut, imgOut, opts.blurIntensity);
  threshold(imgOut, imgOut, opts.minThreshold, opts.maxThreshold, opts.thresholdType);
  // cvtColor(imgOut, imgOut, CV_RGB2GRAY);
}

Mat extractContours(const Mat &imgIn, Options opts, bool debugRecordings)
{
  Mat dst = Mat::zeros(imgIn.size(), CV_8UC1);
  Mat contourDst = Mat::zeros(imgIn.size(), CV_8UC1);
  prepareImage(imgIn, dst, opts);
  if (debugRecordings) cvdbg::recordImage(dst, "prep");

  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  findContours(dst, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));

  if (contours.empty()) return contourDst;

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
  if (debugRecordings) cvdbg::recordImage(contourDst, "contour");

  return contourDst;
}

vector<cv::Vec4i> detectLines(Mat &imgIn, Options opts)
{
  cv::Size size = imgIn.size();
  if (false)
  {
    vector<cv::Vec2f> lines;
    int minLineLength = std::min(size.width, size.height)*.5;
    HoughLines(imgIn, lines, opts.houghRho, opts.houghTheta, opts.houghMinLineLength, opts.houghMinLineGap);

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

vector<cv::Vec4i> findLinesOfLargestRect(Mat &src, Options opts, bool debugRecordings)
{
  Mat contours = extractContours(src, opts, debugRecordings);
  vector<cv::Vec4i> lines = detectLines(contours, opts);
  
  if (debugRecordings)
  {
    for (auto l : lines) {
      line(src, Point(l[0], l[1]), Point(l[2], l[3]), cvhelper::randomColor(), 3, CV_AA);
    }
    cvdbg::recordImage(src, "lines");
  }
  return lines;
}

Mat screenProjection(Mat &src, cv::Size size, Options opts, bool debugRecordings)
{
  auto lines = findLinesOfLargestRect(src, opts, debugRecordings);
  auto from = cv::Rect(cv::Point(0,0), src.size());
  auto into = cv::Rect(cv::Point(0,0), size);
  auto corners = quad::findCorners(lines, from, opts.quadOptions);
  auto tfm = quad::cornerTransform(corners, into, opts.quadOptions);
  return tfm;
}

Mat applyScreenProjection(cv::Mat &in, cv::Mat &projection, cv::Size size, Options opts, bool debugRecordings)
{
  cv::Mat projected = cv::Mat::zeros(size.width, size.height, in.type());
  cv::warpPerspective(in, projected, projection, size);
  if (debugRecordings) cvdbg::recordImage(projected, "projection");
  return projected;
}

Mat extractLargestRectangle(Mat &src, cv::Size size, Options opts, bool debugRecordings)
{
  Mat proj = screenProjection(src, size, opts, debugRecordings);
  return applyScreenProjection(src, proj, size, opts, debugRecordings);
}

quad::Corners cornersOfLargestRect(Mat &src, Options opts, bool debugRecordings)
{
  auto lines = findLinesOfLargestRect(src, opts, debugRecordings);
  auto bounds = cv::Rect(cv::Point(0,0), src.size());
  return quad::findCorners(lines, bounds, opts.quadOptions);
}

} // screen
} // vision
