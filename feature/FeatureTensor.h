#include "tensorflow/core/public/session.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

#include "model.h"

typedef unsigned char uint8;

class FeatureTensor
{
public:
	FeatureTensor(DeepSortPar& par);
	~FeatureTensor();
	bool getRectsFeature(const cv::Mat& img, DETECTIONS& d);
	std::string getErrorInfo(void);

private:
	FeatureTensor& operator = (const FeatureTensor&);
	bool init(DeepSortPar& par);
	void tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf);

private:
	int feature_dim;
	bool m_initStatus;
	std::string m_errMsg;
	tensorflow::Session* session;
	tensorflow::GraphDef graph_def;
	tensorflow::SessionOptions sessOptions;
	std::vector<tensorflow::Tensor> output_tensors;
	std::vector<tensorflow::string> outnames;
	tensorflow::string input_layer;
};

