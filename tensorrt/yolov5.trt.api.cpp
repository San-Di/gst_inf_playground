#include "yolov5.trt.api.h"

#include "yolov5_phrd.h"
#include "yololayerv5.h"

#include <chrono>

inline bool check_exist_file(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}
#if(1)
cv::Rect get_rect(int net_w, int net_h, cv::Size& img_size, float bbox[4]) {
	int l, r, t, b;
	float r_w = net_w / (img_size.width * 1.0);
	float r_h = net_h / (img_size.height * 1.0);
	// if (r_h > r_w) {

	l = bbox[0] - bbox[2] / 2.f;
	r = bbox[0] + bbox[2] / 2.f;
	t = bbox[1] - bbox[3] / 2.f;
	b = bbox[1] + bbox[3] / 2.f;
	l = l / r_w;
	r = r / r_w;
	t = t / r_h;
	b = b / r_h;

	return cv::Rect(l, t, r - l, b - t);
}
float iou(float lbox[4], float rbox[4]) {
	float interBox[] = {
		(std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
		(std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
		(std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
		(std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

bool cmp(const Detection& a, const Detection& b) {
	return a.conf > b.conf;
}

void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh = 0.5) {
	int det_size = sizeof(Detection) / sizeof(float);
	std::map<float, std::vector<Detection>> m;
	for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
		if (output[1 + det_size * i + 4] <= conf_thresh) continue;
		Detection det;
		memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
		m[det.class_id].push_back(det);
	}
	for (auto it = m.begin(); it != m.end(); it++) {
		//std::cout << it->second[0].class_id << " --- " << std::endl;
		auto& dets = it->second;
		std::sort(dets.begin(), dets.end(), cmp);
		for (size_t m = 0; m < dets.size(); ++m) {
			auto& item = dets[m];			
			res.push_back(item);
			for (size_t n = m + 1; n < dets.size(); ++n) {
				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}
}
#endif
struct yolov5_trt_api::implement
{
	yolov5_trt_api* m_parent;
	std::shared_ptr<yolov5_api> net;
		
	std::vector<float> inputData;
	//std::vector<float> outputData;
	std::vector<float> prob;
	int outputCount;

	int m_batch;
	int m_cfgIdx;
	
	//net setting
	int input_width;
	int input_height;
	int num_class;
	RUN_MODE runmode;
	float nmsThresh;

public:
	implement(yolov5_trt_api* parent)
		: m_parent (parent)
		, m_batch(0)
		, m_cfgIdx(0)
		, nmsThresh(0.0)
	{
	};

	~implement()
	{
		if (net)
			net = nullptr;

		inputData.clear();
		//outputData.clear();
		prob.clear();
	}

	void init(int cfgIdx, int yolo_id, int batch)
	{
		m_cfgIdx = cfgIdx;
		m_batch = batch;

		input_width = net_config(m_cfgIdx).get_netw();
		input_height = net_config(m_cfgIdx).get_neth();
		num_class = net_config(m_cfgIdx).get_cls_num();
		nmsThresh = net_config(m_cfgIdx).get_nms_thresh();

		std::string runmode_str = net_config(m_cfgIdx).get_runmode();

		if(runmode_str == "FP32")
			runmode = RUN_MODE::USE_FP32;
		else if (runmode_str == "FP16")
			runmode = RUN_MODE::USE_FP16;
		else
		{
			runmode_str = "FP32";
			runmode = RUN_MODE::USE_FP16;
		}

		std::string config_path = net_config(m_cfgIdx).get_weight_path();
		std::string engine_path = net_config(m_cfgIdx).get_weight_path();

		std::string model_extension = ".wts";
		std::string name_suffix = "_" + runmode_str + "_b" + std::to_string(m_batch) + "_device" + std::to_string(net_config(m_cfgIdx).get_deviceid(yolo_id)) + ".engine";

		size_t start_pos = config_path.find(model_extension);
		if (start_pos == std::string::npos)
			return;
		engine_path.replace(start_pos, model_extension.length(), name_suffix);

		assert(net_config(m_cfgIdx).get_deviceid(yolo_id) >= 0, "Cannot initialize yolo instance for device.");


		std::string yolo_arch = net_config(m_cfgIdx).get_arch();
		if (yolo_arch == "YOLOV5_M")
		{
			net = std::make_shared<yolov5m_trt>();
		}
		else if (yolo_arch == "YOLOV5_X")
		{
			net = std::make_shared<yolov5x_trt>();
		}
		else if (yolo_arch == "YOLOV5_PHRD")
		{
			net = std::make_shared<yolov5phrd_trt>();
		}
		else
		{
			std::cout << "Cannot initialize the architecture of: " << yolo_arch << std::endl;
			exit(-1);
		}

		net->initialization(runmode, m_batch, net_config(m_cfgIdx).get_deviceid(yolo_id), input_width, input_height, num_class);

		if (check_exist_file(engine_path))
		{
			std::cout << "Loading engine: " << engine_path << std::endl;
			net->load_engine(engine_path, net_config(m_cfgIdx).get_deviceid(yolo_id));
		}
		else
		{
			std::cout << "Generating new engine: " << engine_path << std::endl;

			net->create_engine(config_path, engine_path);
			
			net->load_engine(engine_path, net_config(m_cfgIdx).get_deviceid(yolo_id));
		}
		
		//input init
		inputData.reserve(input_height * input_width * 3 * m_batch);

		//output init
		outputCount = net->getOutputSize();
		//outputData.resize(outputCount);
		prob.resize(m_batch * outputCount);
	
	}

	void close()
	{

	}

	void prepareImage(cv::Mat& img, float* data, int cfgIdx)
	{
		using namespace cv;


		int c = 3;
		int h = net_config(cfgIdx).get_neth();   //net h
		int w = net_config(cfgIdx).get_netw();   //net w

		cv::Mat cropped(h, w, CV_8UC3, 127);
		cv::resize(img, cropped, cv::Size(w, h), 0, 0, INTER_CUBIC);
		cv::cvtColor(cropped, cropped, CV_BGR2RGB);

		cv::Mat img_float;
		if (c == 3)
			cropped.convertTo(img_float, CV_32FC3, 1 / 255.0);
		else
			cropped.convertTo(img_float, CV_32FC1, 1 / 255.0);

		//HWC TO CHW
		std::vector<Mat> input_channels(c);
		cv::split(img_float, input_channels);

		int channelLength = h * w;
		for (int i = 0; i < c; ++i) {
			memcpy(data, input_channels.at(i).data, channelLength * sizeof(float));
			data += channelLength;
		}


	}


	void data_preparation(std::vector<cv::Mat> imgs)
	{
		float* data = inputData.data();
		for (auto& img : imgs)
		{			
			prepareImage(img, data, m_cfgIdx);
			data += (input_height * input_width * 3);
		}
	}

	void inference(cv::Mat& img)
	{				
		prepareImage(img, inputData.data(), m_cfgIdx);
		net->doInference(inputData.data(), prob.data(), m_batch);
	}
	void inference(float* feed_data, int batch)
	{
		if(feed_data!= nullptr)
			net->doInference(feed_data, prob.data(), batch);
	}
	
	void get_output(std::vector<cv::Size> img_sizes, std::vector<std::vector<bbox_t>>& v_bbox, float thresh)
	{
		std::vector<std::vector<Detection>> outputBox(img_sizes.size());
				
		for (int b = 0; b < img_sizes.size(); b++) {
			auto& res = outputBox[b];
			nms(res, &prob[b * outputCount], thresh, nmsThresh);

			std::vector<bbox_t> get_bbox;
			for (auto det : outputBox[b])
			{
				cv::Rect temp_rect = get_rect(input_width, input_height, img_sizes.at(b), det.bbox);
				bbox_t temp_bbox;
				temp_bbox.x = temp_rect.x;
				temp_bbox.y = temp_rect.y;
				temp_bbox.w = temp_rect.width;
				temp_bbox.h = temp_rect.height;
				temp_bbox.obj_id = det.class_id;
				temp_bbox.prob = det.conf;
				get_bbox.push_back(temp_bbox);
			}
			v_bbox.push_back(get_bbox);
		}
	}

	//Util functions
	bool is_ready() const
	{
		return net->is_ready();
	}

	int get_netw() const
	{
		return input_width;
	}

	int get_neth() const
	{
		return input_height;
	}
};

yolov5_trt_api::yolov5_trt_api()
	: impl(new implement(this))
{
}
yolov5_trt_api::~yolov5_trt_api()
{
	if (impl)
	{
		delete impl;
		impl = nullptr;
	}
}
void yolov5_trt_api::init(int cfgIdx, int yolo_id, int batch)
{
	impl->init(cfgIdx, yolo_id, batch);
}

bool yolov5_trt_api::is_ready() const
{
	return impl->is_ready();
}
int yolov5_trt_api::get_neth() const
{
	return impl->get_neth();
}
int yolov5_trt_api::get_netw() const
{
	return impl->get_netw();
}
void yolov5_trt_api::inference(float* feed_data, int batch)
{
	impl->inference(feed_data, batch);
}
void yolov5_trt_api::get_output(std::vector<cv::Size> img_size, std::vector<std::vector<bbox_t>>& v_bbox, float thresh)
{
	impl->get_output(img_size, v_bbox, thresh);
}
