#pragma once

#include<vector>
#include <opencv2/opencv.hpp>

#include "yolo.trt.api.base.h"

struct yolov5_trt_api : yolo_trt
{
	yolov5_trt_api();
	~yolov5_trt_api();

	void init(int cfgIdx = 0, int device_id = 0, int batch = 1) override;

	void inference(float* feed_data, int batch) override;	
	void get_output(std::vector<cv::Size>, std::vector<std::vector<bbox_t>>&, float thresh = 0.5) override;

	int get_netw() const override;
	int get_neth() const override;

	bool is_ready() const override;

private:
	struct implement;
	implement* impl;
};