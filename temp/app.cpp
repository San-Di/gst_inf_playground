
#include <gst/gst.h>
#include <iostream>
#include "yolov5_phrd.h"


int main(){

    g_print(">>>>> Test YOLO <<<");
    std::string runmode_str = "FP16";
    RUN_MODE runmode = RUN_MODE::USE_FP16;
    int max_batch = 3;

    int netw = 672;
    int neth = 384;
    int num_class = 2;
    int gpu_idx = 0;

    std::string config_path = "incident_2cls_phrd_1344_exp2_best.wts";
    std::string engine_path = "incident_2cls_phrd_1344_exp2_best.wts";
    std::string model_extension = ".wts";

    std::string name_suffix = "_" + runmode_str + "_b" + std::to_string(max_batch) + "_device" + std::to_string(gpu_idx) + ".engine";

    size_t start_pos = config_path.find(model_extension);

    engine_path.replace(start_pos, model_extension.length(), name_suffix);

    assert(gpu_idx >= 0, "Cannot initialize yolo instance for device.");


    std::string yolo_arch = "YOLOV5_PHRD";
    std::shared_ptr<yolov5phrd_trt> net = std::make_shared<yolov5phrd_trt>();

    net->initialization(runmode, max_batch, gpu_idx, netw, neth, num_class);
    std::cout << "Generating new engine: " << engine_path << std::endl;

    net->create_engine(config_path, engine_path);

    net->load_engine(engine_path, gpu_idx);
    // if (check_exist_file(engine_path))
    // {
    //   std::cout << "Loading engine: " << engine_path << std::endl;
    //   net->load_engine(engine_path, gpu_idx);
    // }
    // else
    // {
    //   std::cout << "Generating new engine: " << engine_path << std::endl;

    //   net->create_engine(config_path, engine_path);

    //   net->load_engine(engine_path, gpu_idx);
    // }

    //input init
    // inputData.reserve(input_height * input_width * 3 * config->max_batch);

    //output init
    int outputCount = net->getOutputSize();
    //outputData.resize(outputCount);
    // prob.resize(config->max_batch * outputCount);
    g_print("OutputSize > %d\n", outputCount);
}