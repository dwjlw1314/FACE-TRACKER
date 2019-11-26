#pragma once
#ifndef DATATYPE_H
#define DATATYPEH

#include <cstddef>
#include <vector>
//#include <Eigen>
#include <Eigen/Core>

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURESS;
//typedef std::vector<FEATURE> FEATURESS;

//Kalmanfilter
//typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

//main
using RESULT_DATA = std::pair<int, DETECTBOX>;
using RETURN_RESULT = std::vector<RESULT_DATA>;

//tracker:
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
typedef struct t{
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
}TRACHER_MATCHD;

//linear_assignment:
typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;

/*
 * version 1.0.0.0
 * used DeepSort and TensorFlow
 *
 * input value
 * DeepSort关键参数:
 * args_nn_budget:
 * args_max_cosine_distance:
 */
typedef struct DeepSortPar
{
	int feature_dim;
	/*
	 * track par
	 */
	int args_nn_budget;
	float args_max_cosine_distance;
	/*
	 * Tensorflow feature par
	 */
	float gpu_memory_fraction;
	const char *tf_model_path;
	const char *input_layer;
	const char *outnames;
} DeepSortPar, *pDeepSortPar;


#endif // DATATYPE_H
