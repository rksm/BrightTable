#ifndef HAND_DETECTOR_SERVER_SERVICES_H_
#define HAND_DETECTOR_SERVER_SERVICES_H_

#include "json/json.h"

#include "l2l.hpp"
typedef std::shared_ptr<l2l::L2lServer> Server;

namespace handdetection {
namespace server {
  
void captureCameraService(Json::Value msg, Server server);
void uploadImageService(Json::Value msg, Server server);
void recognizeScreenCornersService(Json::Value msg, Server server);
void screenCornersTransform(Json::Value msg, Server server);
void screenTransform(Json::Value msg, Server server);
void handDetection(Json::Value msg, Server server);
void handDetectionStreamStart(Json::Value msg, Server server);
void handDetectionStreamStop(Json::Value msg, Server server);

}
}

#endif  // HAND_DETECTOR_SERVER_SERVICES_H_
