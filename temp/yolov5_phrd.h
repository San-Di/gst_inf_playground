#ifndef HEADER_YOLOV5_PHRD_
#define HEADER_YOLOV5_PHRD_

#include "yolov5_base.h"
#include "logging.h"

struct yolov5phrd_trt : yolov5_api
{
public:
	yolov5phrd_trt();
	~yolov5phrd_trt();

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

	int getOutputSize() override;

	bool is_ready() const;

private:
	struct implement;
	implement* impl;
};
#endif
