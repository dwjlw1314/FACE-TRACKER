/*
 * gsFaceTrackSDK.h
 *
 *  Created on: Nov 11, 2019
 *      Author: ai_002
 */

#ifndef GSFACETRACKSDK_H_
#define GSFACETRACKSDK_H_

#include <string>
#include "matching/tracker.h"
#include "feature/FeatureTensor.h"

#ifdef _MSC_VER
#ifndef _EXPORT_LIBXLCRACK_DLL_
	#define EXPORT_LIBXLCRACK  _declspec(dllimport)
#else
	#define EXPORT_LIBXLCRACK  _declspec(dllexport)
#endif
#else
#define GS_VISIBILITY __attribute__ ((visibility ("default")))
#endif

/*!
 * MultiThread support
 * less OPENCV 2.3.14
 * face tracker module; hardware:NVIDIA and less 1.2G GPU memory
 * return face ID values and face coordinate
 */
class gsFaceTrackSDK
{
public:
	/*
	 * args_nn_budget = 100
	 * args_max_cosine_distance = 0.2
	 * default model path "./tensorflow/mars-small128.pb"
	 */
	GS_VISIBILITY gsFaceTrackSDK(pDeepSortPar = nullptr);

	/*
	 * Auto Call
	 */
	~gsFaceTrackSDK();

	/*
	 * v1.0.0.0 更新跟踪计数id，可能会影响跟踪效果，最好在没有人脸的帧进行调用
	 * v1.0.0.1 添加无脸帧的判断
	 */
	GS_VISIBILITY bool recountTrackerId();

    /*!
     * \brief Get Faces Track ID,Rect
     * \param picture data, cv::Mat BGR format
     * \param return result include <trackid,DETECTBOX,landmarks,faceid>;
     * faceid 可以作为人脸检测框和跟踪框的唯一性比对
     * \return version string
     */
	GS_VISIBILITY bool getFacesTrackResult(cv::Mat&, DETECTIONS&, RETURN_RESULT&);

	/*
	 * 年度汇报演示接口，获取tracker是否是剔除状态 UNdefine
	 */
	//GS_VISIBILITY void getFacesTrackStatus(TRACKER_STATUS&);

private:
	int m_time_since_update;
	size_t tracker_num;
	tracker *m_ptracker;
	FeatureTensor *m_pFeatureTensor;
};

#endif /* GSFACETRACKSDK_H_ */
