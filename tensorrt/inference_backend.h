#include <iostream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct Yolo_data {
  cv::Size& im_size;
  float* feed_data;
  int _channel;
  float _threshold;
};

struct Buffer_data {
  float* net_input_buff;
  int m_max_batch;
  std::shared_ptr<yolo_trt> m_yolo;
}

struct bbox_t {
	unsigned int x, y, w, h;	// (x,y) - top-left corner, (w, h) - width & height of bounded box
	float prob;					// confidence - probability that the object was found correctly
	unsigned int obj_id;		// class of object - from range [0, classes-1]	
};

struct yolo_trt
{
	yolo_trt() {};
	~yolo_trt() {};

	virtual void init(int cfgIdx = 0, int device = 0, int batch = 1) = 0;

	virtual void inference(float* feed_data, int batch) = 0;
	virtual void get_output(std::vector<cv::Size>, std::vector<std::vector<bbox_t>>&, float thresh = 0.9) = 0;

	virtual int get_netw() const = 0;
	virtual int get_neth() const = 0;

	virtual bool is_ready() const = 0;
};


struct yolov5_trt_api : yolo_trt
{
	yolov5_trt_api();
	~yolov5_trt_api();

	void init(int cfgIdx = 0, int device_id = 0, int batch = 1) override;
  void load_engine(std::string engine_path);
	void inference(float* feed_data, int batch) override;	
	void get_output(std::vector<cv::Size>, std::vector<std::vector<bbox_t>>&, float thresh = 0.5) override;

	int get_netw() const override;
	int get_neth() const override;

	bool is_ready() const override;

private:
	struct implement;
	implement* impl;
};

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
