/*
 * FeatureTensor.cpp
 *
 *  Created on: Dec 15, 2017
 *      Author: wj
 */

#include "FeatureTensor.h"
using namespace tensorflow;

FeatureTensor::FeatureTensor(DeepSortPar& par)
{
	//prepare model:
	m_initStatus = init(par);
}

FeatureTensor::~FeatureTensor()
{
	if(session != nullptr)
	{
		session->Close();
		delete session;
	}
	output_tensors.clear();
	outnames.clear();
}

bool FeatureTensor::init(DeepSortPar& par)
{
	Status status;
	/*
	//记录设备指派情况
	sessOptions.config.set_log_device_placement(true);
	//自动选择运行设备
	sessOptions.config.set_allow_soft_placement(true);

	//两种控制GPU资源使用的方法，一是让TF在运行过程中动态申请显存;二是限制GPU的使用预加载比例限制#0.5 分配50%
	sessOptions.config.mutable_gpu_options()->set_allow_growth(true);
	*/
	sessOptions.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(par.gpu_memory_fraction);

	session = NewSession(sessOptions);
	if(session == nullptr)
	{
		m_errMsg = "Init session false!!";
		return false;
	}
/*
	Session * session;
	Status status = NewSession(sessOptions,&session);
*/
	const tensorflow::string pathToGraph = par.tf_model_path;

	status = ReadBinaryProto(tensorflow::Env::Default(), pathToGraph, &graph_def);
	if(!status.ok())
	{
		/*
		cout << status_load.ToString() << "\n";
		system("pause");
		*/
		m_errMsg = status.error_message();
		return false;
	}

/*
	for (int i=0; i < graph_def.node_size(); i++)
	{
	    std::string name = graph_def.node(i).name();
	    std::cout << name << std::endl;
	}
*/
	status = session->Create(graph_def);
	if(!status.ok())
	{
		m_errMsg = status.error_message();
		return false;
	}

	feature_dim = par.feature_dim; //128
	input_layer = par.input_layer;//"images"
	outnames.push_back(par.outnames);//"features"
	return true;
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img, DETECTIONS& d)
{
	int i = 0;
	std::vector<cv::Mat> mats;

	if (!m_initStatus)
		return false;

	//empty face box
	if (d.empty()) return true;

	for(DETECTION_ROW& dbox : d)
	{
		cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
				int(dbox.tlwh(2)), int(dbox.tlwh(3)));
		rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
		rc.width = rc.height * 0.5;
		rc.x = (rc.x >= 0 ? rc.x : 0);
		rc.y = (rc.y >= 0 ? rc.y : 0);
		rc.width = (rc.x + rc.width <= img.cols? rc.width: (img.cols-rc.x));
		rc.height = (rc.y + rc.height <= img.rows? rc.height:(img.rows - rc.y));

		cv::Mat mattmp = img(rc).clone();
		cv::resize(mattmp,mattmp, cv::Size(64,128), 0, 0, cv::INTER_AREA);
		mats.push_back(mattmp);
	}
	int count = mats.size();

	Tensor input_tensor(DT_UINT8, TensorShape({count, 128, 64, 3}));
	tobuffer(mats, input_tensor.flat<uint8>().data());
    std::vector<std::pair<string, Tensor> > feed_dict;
    feed_dict.push_back(std::make_pair(input_layer, input_tensor));
//	std::vector<std::pair<tensorflow::string, Tensor>> feed_dict = {
//			{input_layer, input_tensor},
//	};
	Status status = session->Run(feed_dict, outnames, {}, &output_tensors);
	if(!status.ok())
	{
		m_errMsg = status.error_message();
		return false;
	}
	float* tensor_buffer = output_tensors[0].flat<float>().data();
	for(DETECTION_ROW& dbox : d)
	{
		for(int j = 0; j < feature_dim; j++)
			dbox.feature[j] = tensor_buffer[i*feature_dim+j];
		i++;
	}
	return true;
}

std::string FeatureTensor::getErrorInfo(void)
{
	return m_errMsg;
}

void FeatureTensor::tobuffer(const std::vector<cv::Mat> &imgs, uint8 *buf)
{
	int pos = 0;
	for(const cv::Mat& img : imgs)
	{
		int Lenth = img.rows * img.cols * 3;
		int nr = img.rows;
		int nc = img.cols;
		if(img.isContinuous())
		{
			nr = 1;
			nc = Lenth;
		}
		for(int i = 0; i < nr; i++)
		{
			const uchar* inData = img.ptr<uchar>(i);
			for(int j = 0; j < nc; j++)
			{
				buf[pos] = *inData++;
				pos++;
			}
		}//end for
	}//end imgs;
}
