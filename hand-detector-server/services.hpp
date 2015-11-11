#ifndef HAND_DETECTOR_SERVER_SERVICES_H_
#define HAND_DETECTOR_SERVER_SERVICES_H_

#include "json/json.h"

#include "l2l.hpp"
typedef std::shared_ptr<l2l::L2lServer> Server;

namespace handdetection {
namespace server {
  
void captureCameraService(Json::Value msg, Server server);

}
}

#endif  // HAND_DETECTOR_SERVER_SERVICES_H_
