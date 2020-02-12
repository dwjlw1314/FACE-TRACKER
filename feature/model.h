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
 * facepts: fine faces landmarks point
 */
class DETECTION_ROW {
public:
    DETECTBOX tlwh; //float
    float confidence; //float
    FEATURE feature; //float32
#ifdef USE_FACE_DETECT
    size_t faceid;  //face mark_missed use, return to interface
    FacePts facepts; //structure
#endif
    DETECTBOX to_xyah() const;
    DETECTBOX to_tlbr() const;
};

typedef std::vector<DETECTION_ROW> DETECTIONS;

#endif // MODEL_H
