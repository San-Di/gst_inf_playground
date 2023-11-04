#ifndef HEADER_YOLOV5_X_
#define HEADER_YOLOV5_X_

#include "yolov5_base.h"
#include "logging.h"
struct yolov5x_trt : yolov5_api
{
public:
	yolov5x_trt();
	~yolov5x_trt();

	void initialization(
		RUN_MODE mode,
		int maxBatchSize,
		int deviceid,
		int _inSize_w,
		int _inSize_h,
		int _numClass) override;

	void create_engine(
		std::string& wts_name,
		std::string& engine_name) override;

	void load_engine(
		std::string& engine_name,
		int deviceid) override;

	void doInference(float* input, float* output, int batchSize) override;
	void doInference(float* output, int batchSize) override;
	void* get_cuda_stream() override;
	void* get_input_buff() override;

	int getOutputSize() override;

	bool is_ready() const;

private:
	struct implement;
	implement* impl;
};
#endif
