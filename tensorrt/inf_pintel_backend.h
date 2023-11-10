#include <iostream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "yolov5_phrd.h"
/**
 * TODO:: The initialization construction should assign 
 * - max_batch
 * - config_ID
 * - gpu_ID
 * - nms Threshold
 * - num_class
 * - input_width
 * - input_height
 * - runmode <FP32, INT64>
 * from configuration file reading
*/
// cv::Rect get_rect(int net_w, int net_h, cv::Size& img_size, float bbox[4]) {
// 	int l, r, t, b;
// 	float r_w = net_w / (img_size.width * 1.0);
// 	float r_h = net_h / (img_size.height * 1.0);
// 	// if (r_h > r_w) {

// 	l = bbox[0] - bbox[2] / 2.f;
// 	r = bbox[0] + bbox[2] / 2.f;
// 	t = bbox[1] - bbox[3] / 2.f;
// 	b = bbox[1] + bbox[3] / 2.f;
// 	l = l / r_w;
// 	r = r / r_w;
// 	t = t / r_h;
// 	b = b / r_h;

// 	return cv::Rect(l, t, r - l, b - t);
// }
// float iou(float lbox[4], float rbox[4]) {
// 	float interBox[] = {
// 		(std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
// 		(std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
// 		(std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
// 		(std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
// 	};

// 	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
// 		return 0.0f;

// 	float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
// 	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
// }

// bool cmp(const Detection& a, const Detection& b) {
// 	return a.conf > b.conf;
// }

// void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh = 0.5) {
// 	int det_size = sizeof(Detection) / sizeof(float);
// 	std::map<float, std::vector<Detection>> m;
// 	for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
// 		if (output[1 + det_size * i + 4] <= conf_thresh) continue;
// 		Detection det;
// 		memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
// 		if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
// 		m[det.class_id].push_back(det);
// 	}
// 	for (auto it = m.begin(); it != m.end(); it++) {
// 		//std::cout << it->second[0].class_id << " --- " << std::endl;
// 		auto& dets = it->second;
// 		std::sort(dets.begin(), dets.end(), cmp);
// 		for (size_t m = 0; m < dets.size(); ++m) {
// 			auto& item = dets[m];			
// 			res.push_back(item);
// 			for (size_t n = m + 1; n < dets.size(); ++n) {
// 				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
// 					dets.erase(dets.begin() + n);
// 					--n;
// 				}
// 			}
// 		}
// 	}
// }
struct yolo_data
{
  cv::Size& im_size;
  float* feed_data;
  int _channel;
  float _threshold;
};

struct pintel_backend_api {
    pintel_backend_api(){};
    ~pintel_backend_api(){};

    private:
        std::vector<float> input_data;
        std::vector<float> prob;
        int outputCount;
        int m_batch;
        int m_cfgIdx;

        RUN_MODE runmode = RUN_MODE::USE_FP16;
        int max_batch = 3;

        int netw = 672;
        int neth = 384;
        int num_class = 2;
        int gpu_idx = 0;
        std::deque<yolo_data> m_data_depot;
        std::string yolo_arch = "YOLOV5_PHRD";
        std::string engine_path = "D:/workspace/weights/incident_2cls_phrd_1344_exp2_best_FP16_b3_device0.engine"
    public:
        void load_engine(){
                    
            std::shared_ptr<yolov5phrd_trt> net = std::make_shared<yolov5phrd_trt>();

            net->initialization(runmode, max_batch, gpu_idx, netw, neth, num_class);
            net->load_engine(engine_path, gpu_idx);

        }

        void inference(std::vector<cv::Size>, std::vector<std::vector<bbox_t>>&, float thresh = 0.5){
            prepareImage(img, inputData.data(), m_cfgIdx);
		    net->doInference(inputData.data(), prob.data(), m_batch);
            std::cout<<"INFerence is finished successfully*****";
        }

        
        // void get_output(std::vector<cv::Size> img_sizes, std::vector<std::vector<bbox_t>>& v_bbox, float thresh)
        // {
        //     std::vector<std::vector<Detection>> outputBox(img_sizes.size());
                    
        //     for (int b = 0; b < img_sizes.size(); b++) {
        //         auto& res = outputBox[b];
        //         nms(res, &prob[b * outputCount], thresh, nmsThresh);

        //         std::vector<bbox_t> get_bbox;
        //         for (auto det : outputBox[b])
        //         {
        //             cv::Rect temp_rect = get_rect(input_width, input_height, img_sizes.at(b), det.bbox);
        //             bbox_t temp_bbox;
        //             temp_bbox.x = temp_rect.x;
        //             temp_bbox.y = temp_rect.y;
        //             temp_bbox.w = temp_rect.width;
        //             temp_bbox.h = temp_rect.height;
        //             temp_bbox.obj_id = det.class_id;
        //             temp_bbox.prob = det.conf;
        //             get_bbox.push_back(temp_bbox);
        //         }
        //         v_bbox.push_back(get_bbox);
        //     }
        // }




}