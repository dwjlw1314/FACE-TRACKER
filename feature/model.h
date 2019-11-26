#ifndef MODEL_H
#define MODEL_H
#include "dataType.h"
#include <map>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

/**
 * Each rect's data structure.
 * tlwh: topleft point & (w,h)
 * confidence: detection confidence.
 * feature: the rect's 128d feature.
 */
class DETECTION_ROW {
public:
    DETECTBOX tlwh; //np.float
    float confidence; //float
    FEATURE feature; //np.float32
    DETECTBOX to_xyah() const;
    DETECTBOX to_tlbr() const;
};

typedef std::vector<DETECTION_ROW> DETECTIONS;

#endif // MODEL_H
