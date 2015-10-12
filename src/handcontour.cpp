#include "HandDetection.h"
#include <numeric>
#include <algorithm>
#include "timer.h"

using namespace cv;

using PointV = vector<Point>;

const Scalar RED   = Scalar(0, 255,   0);
const Scalar GREEN = Scalar(0,   255, 0);
const Scalar BLUE  = Scalar(0,   0,   255);
const Scalar WHITE = Scalar(255, 255, 255);
const Scalar BLACK = Scalar(0,   0,   0);

int lowThreshold = 50;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

struct ConvexityDefectData {
    Point onHullStart;
    Point onHullEnd;
    Point defect;
    float distFromHull;
    float distFromCenter;
};

struct HandData {
    int radius;
    Point center;
    RotatedRect outerBounds;
    ConvexityDefectData convexityDefects;
};

Mat prepareForContourDetection(Mat src)
{

    Mat detected_edges;

    cvtColor(src, detected_edges,CV_RGB2GRAY);

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    // blur(src_gray, detected_edges, Size(4,4));
    // equalizeHist(detected_edges, detected_edges);
    medianBlur(detected_edges, detected_edges, 69);
    recordImage(detected_edges, "blur");

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    threshold(detected_edges, detected_edges, 100, 255, CV_THRESH_BINARY);
    // adaptiveThreshold(detected_edges, detected_edges, 115, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -3);
    // adaptiveThreshold(detected_edges, detected_edges, 165, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 9, -3);
    // recordImage(detected_edges, "threshold");

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    dilate(detected_edges, detected_edges, Mat(3,3, 0), Point(-1, -1), 5);
    recordImage(detected_edges, "dilate");

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    /// Canny detector
    // Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
    // recordImage(detected_edges, "canny");


    return detected_edges;
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void contourHullExtraction(PointV &contours, PointV &hullPoints, vector<int> &hullInts, vector<Vec4i> &defects) {
    convexHull(contours, hullPoints);
    convexHull(contours, hullInts);
    if (hullPoints.size() >= 3) {
        convexityDefects(contours, hullInts, defects);
    }
}

bool contourMoments(PointV &contours, vector<Moments> &momentsVec) {
    Moments mo = moments(contours, false);
    if (mo.m00 > 0) { // self intersecting...
        // Point pos = Point((int) (mo.m10 / mo.m00), (int) (mo.m01 / mo.m00));
        // std::cout
        //     << "Moments: [" << i << "] "
        //     << mo.m10 << "/" << mo.m00 << " = " << pos.x << ","
        //     << mo.m01 << "/" << mo.m00 << " = " << pos.y
        //     << std::endl;
        momentsVec.push_back(mo);
        return true;
    }
    return false;
}

void contourBounds(PointV &contours, Point &center, int &radius, RotatedRect &bounds, RotatedRect &outerBounds, RotatedRect &innerBounds) {
    outerBounds = fitEllipse(contours);
    Size size = outerBounds.size;
    radius = min(size.width, size.height)/2;

    // auto hullEllipse = fitEllipse(hullPoints);
    // ellipse(drawing, hullEllipse, color, 2, 8 );
    bounds = minAreaRect(contours);
    center = bounds.center;
    innerBounds = RotatedRect(
        bounds.center,
        Size2f(bounds.size.width*0.66, bounds.size.height*0.66),
        bounds.angle);
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void drawHull(Mat &drawing, Scalar color, PointV &hullPoints) {
    for (auto hullp : hullPoints) {
        circle(drawing, hullp,   4, Scalar(255,255,0), 2 );
    }
}

void drawMoments(Mat &drawing, Scalar color, vector<Moments> momentsVec) {
    Moments mo = momentsVec.back();
    Point pos = Point((int) (mo.m10 / mo.m00), (int) (mo.m01 / mo.m00));
    circle(drawing, pos, 10, color, 5);
}

void drawConvexityDefects(Mat &drawing, PointV &contours, Scalar color, int radius, Point center, vector<Vec4i> &defects) {
    for (Vec4i d : defects) {
        int startidx = d[0];
        Point ptStart(contours[startidx]); // point of the contour where the defect begins
        int endidx=d[1];
        Point ptEnd(contours[endidx]); // point of the contour where the defect ends
        int faridx=d[2];
        Point ptFar(contours[faridx]);// the farthest from the convex hull point within the defect
        float depth = d[3] / 256; // distance between the farthest point and the convex hull
        if (depth > 20)
        {
            // std::cout << "start" << ptStart << "end" << ptEnd << "far" << ptFar << "depth" << depth << std::endl;
            // Scalar color = CV_RGB(0,255,0);
            // line(drawing, ptStart, ptFar, color, 2 );

            // if (innerBounds.boundingRect().contains(ptFar)) {
            if (norm(ptFar - center) < radius*.8) {

                // line(drawing, ptStart, ptEnd, CV_RGB(255,0,0), 2 );
                line(drawing, ptStart, ptFar, CV_RGB(255,0,0), 6);
                line(drawing, ptEnd, ptFar, CV_RGB(255,0,0), 6);
                // Point mid = (ptStart+ptEnd)*.5;
                // line(drawing, mid, ptFar, CV_RGB(255,0,0), 2 );
                // line(drawing, ptEnd, ptFar, color, 2 );
                // circle(drawing, ptStart,   4, Scalar(100,0,255), 2 );
                circle(drawing, ptFar,   18, CV_RGB(255,0,0), 10 );
            }

        }
    }
}

Mat findContours(Mat src)
{
    Mat dst;
    vector<Vec4i> hierarchy;
    vector<PointV > contours;
    findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));
    vector<PointV> hullsP (contours.size());
    vector<cv::vector<int>> hullsI(contours.size());
    vector<vector<Vec4i> > defects(contours.size());
    vector<vector<Moments>> momentsVec(contours.size());

    // std::cout << "Found " << contours.size() << " contours" << std::endl;
    /// Draw contours
    Mat drawing = Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i< contours.size(); i++)
    {
        double area = std::abs(contourArea(contours[i], true));

        if ((area > 200)) {
            
            contourHullExtraction(contours[i], hullsP[i], hullsI[i], defects[i]);
            RotatedRect outerBounds, bounds, innerBounds; int radius; Point center;
            contourBounds(contours[i], center, radius, bounds, outerBounds, innerBounds);
            bool momentsFound = contourMoments(contours[i], momentsVec[i]);
        
            auto color = randomColor();
            drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
            drawContours(drawing, hullsP, i, color, 1, 8, vector<Vec4i>(), 0, Point());
            drawHull(drawing, color, hullsP[i]);
            if (momentsFound) drawMoments(drawing, color, momentsVec[i]);
            ellipse(drawing, outerBounds, CV_RGB(255,0,0), 2, 8 );
            ellipse(drawing, bounds, CV_RGB(0,255,0), 3, 8 );
            ellipse(drawing, innerBounds, CV_RGB(0,0,255), 4, 8 );
            drawConvexityDefects(drawing, contours[i], color, radius, center, defects[i]);
        }
    }

    recordImage(drawing.clone(), "contour");

    return drawing;
}


// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

const int HIGH = 150;
const int LOW = 150;

Mat processFrame(Mat frameIn) {

    Mat result;

    std::cout << timeToRunMs([&](){
        Mat resized = resizeToFit(frameIn, 1000, 1000);
        recordImage(resized.clone(), "orig");
        result = findContours(prepareForContourDetection(resized));
    }).count() << std::endl;
    // std::cout << d.count() << std::endl;

    return result;
}
