#include "screen-detection.hpp"
#include "quad-transform.hpp"
#include "HandDetection.hpp"

#include <numeric>
#include <algorithm>

using std::vector;
using cv::Mat;
using cv::Point;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void prepareImage(Mat &imgIn, Mat &imgOut)
{
  if (imgIn.type() == CV_8UC3) cvtColor(imgIn, imgOut, CV_BGR2GRAY);
  medianBlur(imgOut, imgOut, 21);
  threshold(imgOut, imgOut, 150, 255, CV_THRESH_BINARY);
}

void extractContours(Mat &imgIn, Mat &imgOut, vector<cv::Point2f> &corners)
{
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  findContours(imgIn, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));

  double area = 0;
  int biggestI = 0;
  for (int i = 0; i < contours.size(); i++) {
    auto areaNow = contourArea(contours[i]);
    if (areaNow > area) {
      biggestI = i;
      area = areaNow;
    }
  }

  if (true) // hull points
  {
    vector<Point> hullPoints;
    convexHull(contours[biggestI], hullPoints);
    drawPointsConnected<Point>(hullPoints, imgOut);

    // // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    // Mat area = Mat::zeros(imgOut.size(), CV_8UC1);
    // cv::RotatedRect r = cv::minAreaRect(hullPoints);
    // cv::Point2f pts[4];
    // r.points(pts);
    // // vector<cv::Point2f> v(pts);
    // vector<cv::Point2f> v(pts, pts + sizeof pts / sizeof pts[0]);
    // drawPointsConnected<cv::Point2f>(v, area);
    // recordImage(area, "contours");

  }

  if (false)
  {
    drawContours(imgOut, contours, biggestI, cv::Scalar(255, 255, 255), 2, 8, hierarchy, 0, Point());
  }

  if (false)
  {
    vector<Point> contours_poly;
    approxPolyDP( Mat(contours[biggestI]), contours_poly, 3, true);
    drawPointsConnected<Point>(contours_poly, imgOut);
  }

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


vector<cv::Vec4i> extractLinesForLargestRectangle(Mat src, bool debugRecordings)
{
  Mat dst = Mat::zeros(src.size(), CV_8UC1);
  prepareImage(src, dst);
  if (debugRecordings) recordImage(dst, "prep");

  Mat contours = Mat::zeros(src.size(), CV_8UC1);
  vector<cv::Point2f> corners;
  extractContours(dst, contours, corners);
  if (debugRecordings) recordImage(contours, "contours");

  vector<cv::Vec4i> lines = detectLines(contours);
  
  if (debugRecordings)
  {
    for (auto l : lines) {
      line(src, Point(l[0], l[1]), Point(l[2], l[3]), randomColor(), 3, CV_AA);
    }
    recordImage(src, "lines");
  }
  return lines;
}

Mat extractLargestRectangle(Mat src, cv::Size size, bool debugRecordings)
{
  vector<cv::Vec4i> lines = extractLinesForLargestRectangle(src, debugRecordings);

  cv::Mat projected = cv::Mat::zeros(size.width, size.height, CV_8UC3);
  projectLinesInto(lines, src, projected);
  if (debugRecordings) recordImage(projected, "projection");
  return projected;
}

Mat screenProjection(Mat src, cv::Size size, bool debugRecordings)
{
  vector<cv::Vec4i> lines = extractLinesForLargestRectangle(src, debugRecordings);
  return projectLinesTransform(lines,
    cv::Rect(cv::Point(0,0), src.size()),
    cv::Rect(cv::Point(0,0), size));
}
