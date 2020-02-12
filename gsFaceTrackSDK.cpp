/*
 * gsFaceTrackSDK.cpp
 *
 *  Created on: Nov 11, 2019
 *      Author: ai_002
 */
#include "gsFaceTrackSDK.h"

using namespace cv;
using namespace std;

gsFaceTrackSDK::gsFaceTrackSDK(pDeepSortPar param)
	:tracker_num(0)
{
	//init default par member
	DeepSortPar default_par {128, 100, 0, 0.2, 0.02,
			     "./tensorflow/mars-small128.pb",
				 "images", "features"};
	if (!param)
		param = &default_par;

	m_time_since_update = param->time_since_update;
	m_ptracker = new tracker(param->args_max_cosine_distance, param->args_nn_budget);
	m_pFeatureTensor = new FeatureTensor(*param);
}

gsFaceTrackSDK::~gsFaceTrackSDK()
{
	delete m_ptracker;
	delete m_pFeatureTensor;
}

bool gsFaceTrackSDK::getFacesTrackResult(Mat& frame, DETECTIONS& detections, RETURN_RESULT& result)
{
	//TENSORFLOW get rect's feature.
	if(!m_pFeatureTensor->getRectsFeature(frame, detections))
	{
		return false;
	}

	m_ptracker->predict();
	m_ptracker->update(detections);

	for(Track& track : m_ptracker->tracks)
	{
		RESULT_DATA data;
		//time_since_update 预测次数
		if(!track.is_confirmed() || track.time_since_update > m_time_since_update)
			continue;
		data.box = std::make_pair(track.track_id, track.to_tlwh());

#ifdef USE_FACE_DETECT
		data.pts = track.m_facepts;
		data.faceid = track.m_faceid;
#endif
		result.push_back(data);
	}
//	char showMsg[10];
//	for(unsigned int k = 0; k < detections.size(); k++)
//	{
//		DETECTBOX tmpbox = detections[k].tlwh;
//		Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
//		rectangle(frame, rect, Scalar(0,0,255), 4);
//	}
//
//	for(unsigned int k = 0; k < result.size(); k++)
//	{
//		DETECTBOX tmp = result[k].second;
//		Rect rect = Rect(tmp(0), tmp(1), tmp(2), tmp(3));
//		rectangle(frame, rect, Scalar(255, 255, 0), 2);
//		sprintf(showMsg, "%d", result[k].first);
//		putText(frame, showMsg, Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
//	}
	tracker_num = result.size();
	return true;
}

bool gsFaceTrackSDK::recountTrackerId()
{
	if (0 == tracker_num)
	{
		m_ptracker->recountTrackerIdx();
		return true;
	}
	else
		return false;
}
