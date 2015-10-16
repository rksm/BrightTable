#include <iostream>
#include "ps3eye.h"
#include <opencv2/opencv.hpp>

#define _saturate(v) static_cast<uint8_t>(static_cast<uint32_t>(v) <= 0xff ? v : v > 0 ? 0xff : 0)

static void yuv422_to_gray(const uint8_t *yuv_src, const int stride, uint8_t *dst, const int width, const int height)
{
    const int yIdx = 0;
    int j, i;
    
    for (j = 0; j < height; j++, yuv_src += stride)
    {
        uint8_t* row = dst + (width * 1) * j; // 4 channels
        
        for (i = 0; i < 2*width; i += 4, row += 2)
        {
            row[0] = _saturate(yuv_src[i + yIdx]);
            row[1] = _saturate(yuv_src[i + yIdx + 2]);
        }
    }
}

#define SAT(c) if (c & (~255)) { if (c < 0) c = 0; else c = 255; }
// #define SAT(c) c;
void uyvy2rgb(int width, int height, unsigned char *src, unsigned char *dest) {

    // videoFrame->GetBytes((void**)&src);

  // for(int i = 0, j=0; i < width * height * 3; i+=6, j+=4)
  // {
  //     dest[i] =   src[j] + src[j+3]*((1 - 0.299)/0.615);
  //     dest[i+1] = src[j] - src[j+1]*((0.114*(1-0.114))/(0.436*0.587)) - src[j+3]*((0.299*(1 - 0.299))/(0.615*0.587));
  //     dest[i+2] = src[j] + src[j+1]*((1 - 0.114)/0.436);
  //     dest[i+3] = src[j+2] + src[j+3]*((1 - 0.299)/0.615);
  //     dest[i+4] = src[j+2] - src[j+1]*((0.114*(1-0.114))/(0.436*0.587)) - src[j+3]*((0.299*(1 - 0.299))/(0.615*0.587));
  //     dest[i+5] = src[j+2] + src[j+1]*((1 - 0.114)/0.436);
  // }

    // for(int i = 0; i < width * height * 3; i=i+3)
    // {
    //     dest[i] = src[i] + src[i+2]*((1 - 0.299)/0.615);
    //     dest[i+1] = src[i] - src[i+1]*((0.114*(1-0.114))/(0.436*0.587)) - src[i+2]*((0.299*(1 - 0.299))/(0.615*0.587));
    //     dest[i+2] = src[i] + src[i+1]*((1 - 0.114)/0.436);
    // }


	int R,G,B;
	int Y1,Y2;
	int cG,cR,cB;

	for(int i=height*width/2;i>0;i--) {
		cB = ((*src - 128) * 454) >> 8;
		cG = (*src++ - 128) * 88;
		Y1 = *src++;
		cR = ((*src - 128) * 359) >> 8;
		cG = (cG + (*src++ - 128) * 183) >> 8;
		Y2 = *src++;

		R = Y1 + cR;
		G = Y1 + cG;
		B = Y1 + cB;

		SAT(R);
		SAT(G);
		SAT(B);

		*dest++ = B;
		*dest++ = G;
		*dest++ = R;

		R = Y2 + cR;
		G = Y2 + cG;
		B = Y2 + cB;

		SAT(R);
		SAT(G);
		SAT(B);

		*dest++ = B;
		*dest++ = G;
		*dest++ = R;
	}
}

unsigned char* getFrame(ps3eye::PS3EYECam::PS3EYERef eye, unsigned char* cam_buffer, uint h, uint w, bool color) {
  using namespace ps3eye;
    
  while (!eye->isNewFrame()) {
    PS3EYECam::updateDevices();
  }
    
  if (color) {
    uyvy2rgb(w, h, (unsigned char *) eye->getLastFramePointer(), cam_buffer);
  } else {
    yuv422_to_gray((unsigned char *) eye->getLastFramePointer(), eye->getRowBytes(), cam_buffer, w, h);
  }
      
  return cam_buffer;
}


int main(int argc, char** argv)
{
  using namespace ps3eye;
  std::vector<PS3EYECam::PS3EYERef> devices = PS3EYECam::getDevices();
  std::cout << devices.size() << std::endl;

  if (devices.empty()) {
    std::cout << "no eye found!" << std::endl;
    return 0;
  }

  bool color = true;
  auto eye = devices.at(0);
  bool res = eye->init(640, 480, 60);
  std::cout << "init eye result " << res << std::endl;
  eye->start();

  int matType = color ? CV_8UC3 : CV_8UC1; // 8 bpp per channel, 1 or 3 channels
  auto w = eye->getWidth(), h = eye->getHeight();
  void *pData = new uint8_t[w*h*4];

  for (;;) {
    // pData = (void*)eye->getLastFramePointer();
    getFrame(eye, (unsigned char*)pData, h, w, color);
    cv::Mat mat(h, w, matType, pData);
    cv::imshow("test", mat);
    if (cv::waitKey(30) > 1) break;
  }

  std::cout << "Hello world!" << std::endl;
  return 0;
}
