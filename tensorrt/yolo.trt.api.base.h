#pragma once

#include<vector>
#include <opencv2/opencv.hpp>

#ifndef STRUCT_BBOX
#define STRUCT_BBOX
struct bbox_t {
	unsigned int x, y, w, h;	// (x,y) - top-left corner, (w, h) - width & height of bounded box
	float prob;					// confidence - probability that the object was found correctly
	unsigned int obj_id;		// class of object - from range [0, classes-1]	
};
#endif

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