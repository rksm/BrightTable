#include <iostream>
#include <string>
#include <chrono>
#include <thread>

using std::string;

#include "json/json.h"

#include "l2l.hpp"

#include "BoundedBuffer.hpp"

#include "services.hpp"

void startServer(string host, int port, string id)
{

  auto server = l2l::startServer(host, port, id, l2l::Services {
    l2l::createLambdaService("capture-camera", handdetection::server::captureCameraService),
    l2l::createLambdaService("upload-image", handdetection::server::uploadImageService),
    l2l::createLambdaService("recognize-screen-corners", handdetection::server::recognizeScreenCornersService),
    l2l::createLambdaService("screen-corner-transform", handdetection::server::screenCornersTransform),
    l2l::createLambdaService("screen-transform", handdetection::server::screenTransform),
    l2l::createLambdaService("hand-detection", handdetection::server::handDetection),
    l2l::createLambdaService("hand-detection-stream-start", handdetection::server::handDetectionStreamStart),
    l2l::createLambdaService("hand-detection-stream-stop", handdetection::server::handDetectionStreamStop)
  });
  server->debug = true;
}

int main(int argc, char** argv)
{

  int port = 10501;
  std::string host = "0.0.0.0";
  std::string id = "hand-detector-server";
 
 
  startServer(host, port, id);
  // std::thread serverThread(bind(startServer, host, port, id));
  // readFrame();
  // serverThread.join();

  return 0;
}
