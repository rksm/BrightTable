#include "vision/cv-helper.hpp"
#include <numeric>
#include <algorithm>

namespace cvhelper
{

using namespace cv;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// colors

RNG rng(12345);
const Scalar randomColor() {
    return Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
}

void resizeToFit(Mat &in, Mat &out, float maxWidth, float maxHeight)
{
  cv::Size size = in.size();
  if (size.width <= maxWidth && size.height <= maxHeight)
  { if (&in != &out) out = in; return; }

  float h = size.height, w = size.width;
  if (h > maxHeight) {
    w = round(w * (maxHeight / h));
    h = maxHeight;
  }
  if (w > maxWidth) {
    h = round(h * (maxWidth / w));
    w = maxWidth;
  }
  resize(in, out, Size(w,h));
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

template<typename T>
void sortedValues(Mat &in, vector<T> &out)
{
  auto size = in.size();
  for (int i = 0; i < size.height; i++)
    for (int j = 0; j < size.width; j++)
      out[i*size.width + j] = in.at<T>(i, j);
  std::sort(out.begin(), out.end());
}

template<typename T>
std::pair<T,T> percentiles(Mat &in, float percentile=5.0f) {
  
  size_t n = in.cols * in.rows;
  vector<T> sorted(n);
  sortedValues(in, sorted);
  
  size_t idx = std::ceil(percentile/100.0f * (float)n);
  auto low = sorted[idx];
  auto high = sorted[n - idx];

  std::cout << "percentile: " << percentile
            << " n: " << n
            << " low: " << low
            << " high: " << high
            << std::endl;

  return std::make_pair(low, high);
}

void convertToProperGrayscale(Mat &grayMat, float percentile)
{
  // float high = 1.0f, low = 0.0f;
  // if (percentile != 0.0f) {
    auto perc = percentiles<float>(grayMat, percentile);
    float high = perc.second, low = perc.first;
  // }

  for (int i = 0; i < grayMat.rows; i++)
    for (int j = 0; j < grayMat.cols; j++)
    {
      auto val = grayMat.at<float>(i,j);
      grayMat.at<float>(i,j) = std::max(0.0f, std::min(1.0f, 1.0f/(high-low) * (val-low)));
    }
  // float multiplier = 1 / perc.second;
  // grayMat = (grayMat - perc.first) * multiplier;
}

}
