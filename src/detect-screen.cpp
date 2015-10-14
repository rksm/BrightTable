#include "HandDetection.hpp"
#include "quad-transform.h"

#include <numeric>
#include <algorithm>

using std::vector;
using cv::Mat;
using cv::Point;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void prepareImage(Mat &imgIn, Mat &imgOut)
{
  cvtColor(imgIn, imgOut, CV_RGB2GRAY);
  medianBlur(imgOut, imgOut, 21);
  threshold(imgOut, imgOut, 150, 255, CV_THRESH_BINARY);
}

void extractContours(Mat &imgIn, Mat &imgOut)
{
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  findContours(imgIn, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));

  for (int i = 0; i < contours.size(); i++) {
    auto area = contourArea(contours[i]);
    if (area < 100) continue;

    if (true) // hull points
    {
      vector<Point> hullPoints;
      convexHull(contours[i], hullPoints);
      drawPointsConnected<Point>(hullPoints, imgOut);
    }

    if (false)
    {
      drawContours(imgOut, contours, i, cv::Scalar(255, 255, 255), 2, 8, hierarchy, 0, Point());
    }

    if (false)
    {
      vector<Point> contours_poly;
      approxPolyDP( Mat(contours[i]), contours_poly, 3, true);
      drawPointsConnected<Point>(contours_poly, imgOut);
    }
  }
}

vector<cv::Vec4i> detectLines(Mat &imgIn)
{

  if (false)
  {
    vector<cv::Vec2f> lines;
    HoughLines(imgIn, lines, 1, CV_PI/180, 90, 400, 20);

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
      // line(src, pt1, pt2, cv::Scalar(0,0,255), 3, CV_AA);
    }
  }

  if (true)
  {
    cv::Size size = imgIn.size();
    int minLineLength = std::min(size.width, size.height)*.6;
    vector<cv::Vec4i> lines;
    cv::HoughLinesP(imgIn, lines, .5, CV_PI/180, 70, minLineLength, 40);
    return lines;
  }
}


Mat extractLargestRectangle(Mat src, cv::Size size, bool debugRecordings)
{
  Mat dst = Mat::zeros(src.size(), CV_8UC1);
  Mat contours = Mat::zeros(src.size(), CV_8UC1);
  cv::Mat projected = cv::Mat::zeros(size.width, size.height, CV_8UC3);

  prepareImage(src, dst);
  if (debugRecordings) recordImage(dst, "prep");

  extractContours(dst, contours);
  if (debugRecordings) recordImage(contours, "contours");

  vector<cv::Vec4i> lines = detectLines(contours);
  if (debugRecordings) 
  {
    for (auto l : lines) {
      line(src, Point(l[0], l[1]), Point(l[2], l[3]), randomColor(), 3, CV_AA);
    }
    recordImage(src, "lines");
  }
  
  projectLinesInto(lines, src, projected);
  if (debugRecordings) recordImage(projected, "projection");

  return projected;
}
