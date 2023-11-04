#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include "common.hpp"
// ------- Instead common.hpp -----
// #include <fstream>
// #include <map>
// #include <sstream>
// #include <vector>
// #include "yololayerv5.h"
// #include <opencv2/opencv.hpp>
// #include "NvInfer.h"
// #include "NvInferImpl.h"
// #include "NvInferRuntimeCommon.h"
// #include "NvInferRuntime.h"

//-----------------------

#include "yolov5_x.h"


using namespace nvinfer1;
const int OUTPUT_SIZE_YOLOV5X = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME_YOLOV5X = "data";
const char* OUTPUT_BLOB_NAME_YOLOV5X = "prob";

struct yolov5x_trt::implement
{
private:
    RUN_MODE	mode;
    int			maxBatchSize;
    int			deviceid;
    int			inSize_w;
    int			inSize_h;
    int			numClass;

    std::mutex						buff_mutex;
    nvinfer1::IExecutionContext*    mTrtContext;
    nvinfer1::ICudaEngine*          mTrtEngine;
    nvinfer1::IRuntime*             mTrtRunTime;
    cudaStream_t					stream;
    void*                           buffers[2];
    int								inputIndex;
    int								outputIndex;
    Logger                          gLogger;

    bool net_ready;

public:
    implement()
        : net_ready(false)
    {
        std::cout << "YOLOv5X 4-stage architecture loaded.\n";
    };

    ~implement()
    {

#if(1) //error in release resource
        // Release stream and buffers
        std::lock_guard<std::mutex> lk(buff_mutex);
        // Destroy the engine
        mTrtContext->destroy();
        mTrtEngine->destroy();
        mTrtRunTime->destroy();
        // Destroy the buffers
        if (buffers[0])
            CUDA_CHECK(cudaFree(buffers[0]));
        if (buffers[1])
            CUDA_CHECK(cudaFree(buffers[1]));
        cudaStreamDestroy(stream);
#endif
    }


    int get_width(int x, float gw, int divisor) {
        //return math.ceil(x / divisor) * divisor
        if (int(x * gw) % divisor == 0) {
            return int(x * gw);
        }
        return (int(x * gw / divisor) + 1) * divisor;
    }

    int get_width_x(int x, float gw, int divisor = 8) {
        return int(x * 2.5);
    }

    int get_width_cblock(int x, float gw, int divisor) {
        return int(x * 1.5);
    }

    int get_depth(int x, float gd) {
        if (x == 1) {
            return 1;
        }
        else {
            return round(x * gd) > 1 ? round(x * gd) : 1;
        }
    }

    ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {

        #if(USE_YOLOV5_CF5_X_CBLOCK_4STAGE_INCIDENT)
            INetworkDefinition* network = builder->createNetworkV2(0U);
            // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
            ITensor* data = network->addInput(INPUT_BLOB_NAME_YOLOV5X, dt, Dims3{ 3, inSize_h, inSize_w });
            assert(data);

            std::map<std::string, Weights> weightMap = loadWeights_yolo(wts_name);

            /* ------ yolov5 backbone------ */
            auto focus0 = focus(network, weightMap, *data, 3, get_width_x(32, gw), 3, "model.0", inSize_w, inSize_h, numClass);

            auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width_x(64, gw), 3, 2, 1, "model.1");

            // std::cout<<"======== 2 "<<conv1->getOutput(0)->getDimensions().d[0]<<" "<<conv1->getOutput(0)->getDimensions().d[1]<<" "<<conv1->getOutput(0)->getDimensions().d[2]<<std::endl;


            auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width_x(64, gw), get_width_x(64, gw), get_depth(3, gd), true, 1, 0.5, "model.2");

            auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width_x(128, gw), 3, 2, 1, "model.3");
            auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width_x(128, gw), get_width_x(128, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
            auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width_x(256, gw), 3, 2, 1, "model.5");
            auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width_x(256, gw), get_width_x(256, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
            // auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(512, gw), 3, 2, 1, "model.7");
            // auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(512, gw), get_width(512, gw), 5, 9, 13, "model.8");
            // std::cout<<"======== 3 "<<bottleneck_csp6->getOutput(0)->getDimensions().d[0]<<" "<<bottleneck_csp6->getOutput(0)->getDimensions().d[1]<<" "<<bottleneck_csp6->getOutput(0)->getDimensions().d[2]<<std::endl;

            auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width_x(384, gw), 3, 2, 1, "model.7");
            auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width_x(384, gw), get_width_x(384, gw), 5, 9, 13, "model.8");

            // std::cout<<"======== 4 "<<spp8->getOutput(0)->getDimensions().d[0]<<" "<<spp8->getOutput(0)->getDimensions().d[1]<<" "<<spp8->getOutput(0)->getDimensions().d[2]<<std::endl;

            /* ------ yolov5 head ------ */
            auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width_x(384, gw), get_width_x(384, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
            auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width_x(256, gw), 1, 1, 1, "model.10");

            auto upsample11 = network->addResize(*conv10->getOutput(0));
            assert(upsample11);
            upsample11->setResizeMode(ResizeMode::kNEAREST);
            upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

            ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
            auto cat12 = network->addConcatenation(inputTensors12, 2);
            auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width_x(384, gw), get_width_x(256, gw), get_depth(3, gd), false, 1, 0.5, "model.13");

            auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width_x(128, gw), 1, 1, 1, "model.14");
            auto upsample15 = network->addResize(*conv14->getOutput(0));
            assert(upsample15);
            upsample15->setResizeMode(ResizeMode::kNEAREST);
            upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

            ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
            auto cat16 = network->addConcatenation(inputTensors16, 2);

            auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width_x(256, gw), get_width_x(128, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

            // std::cout<<"======== 17 "<<bottleneck_csp17->getOutput(0)->getDimensions().d[0]<<" "<<bottleneck_csp17->getOutput(0)->getDimensions().d[1]<<" "<<bottleneck_csp17->getOutput(0)->getDimensions().d[2]<<std::endl;

            //nvnn
            auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width_x(64, gw), 1, 1, 1, "model.18");
            auto upsample19 = network->addResize(*conv18->getOutput(0));
            assert(upsample19);
            upsample19->setResizeMode(ResizeMode::kNEAREST);
            upsample19->setOutputDimensions(bottleneck_CSP2->getOutput(0)->getDimensions());

            ITensor* inputTensors20[] = { upsample19->getOutput(0), bottleneck_CSP2->getOutput(0) };
            auto cat20 = network->addConcatenation(inputTensors20, 2);

            auto bottleneck_csp21 = C3(network, weightMap, *cat20->getOutput(0), get_width_x(64, gw), get_width_x(64, gw), get_depth(3, gd), false, 1, 0.5, "model.21");

            /* ------ detect ------ */
            // IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp21->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.31.m.0.weight"], weightMap["model.31.m.0.bias"]);
            auto conv22 = convBlock(network, weightMap, *bottleneck_csp21->getOutput(0), get_width_x(64, gw), 3, 2, 1, "model.22");

            ITensor* inputTensors23[] = { conv22->getOutput(0), conv18->getOutput(0) };
            auto cat23 = network->addConcatenation(inputTensors23, 2);

            auto bottleneck_csp24 = C3(network, weightMap, *cat23->getOutput(0), get_width_x(128, gw), get_width_x(128, gw), get_depth(3, gd), false, 1, 0.5, "model.24");

            IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp24->getOutput(0), 3 * (numClass + 5), DimsHW{ 1, 1 }, weightMap["model.31.m.1.weight"], weightMap["model.31.m.1.bias"]);

            auto conv25 = convBlock(network, weightMap, *bottleneck_csp24->getOutput(0), get_width_x(128, gw), 3, 2, 1, "model.25");
            ITensor* inputTensors26[] = { conv25->getOutput(0), conv14->getOutput(0) };
            auto cat26 = network->addConcatenation(inputTensors26, 2);
            auto bottleneck_csp27 = C3(network, weightMap, *cat26->getOutput(0), get_width_x(256, gw), get_width_x(256, gw), get_depth(3, gd), false, 1, 0.5, "model.27");
            IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp27->getOutput(0), 3 * (numClass + 5), DimsHW{ 1, 1 }, weightMap["model.31.m.2.weight"], weightMap["model.31.m.2.bias"]);

            auto conv28 = convBlock(network, weightMap, *bottleneck_csp27->getOutput(0), get_width_x(256, gw), 3, 2, 1, "model.28");
            ITensor* inputTensors29[] = { conv28->getOutput(0), conv10->getOutput(0) };
            auto cat29 = network->addConcatenation(inputTensors29, 2);
            auto bottleneck_csp30 = C3(network, weightMap, *cat29->getOutput(0), get_width_x(384, gw), get_width_x(384, gw), get_depth(3, gd), false, 1, 0.5, "model.30");

            IConvolutionLayer* det3 = network->addConvolutionNd(*bottleneck_csp30->getOutput(0), 3 * (numClass + 5), DimsHW{ 1, 1 }, weightMap["model.31.m.3.weight"], weightMap["model.31.m.3.bias"]);

            IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp21->getOutput(0), 3 * (numClass + 5), DimsHW{ 1, 1 }, weightMap["model.31.m.0.weight"], weightMap["model.31.m.0.bias"]);
            auto yolo = addYoLoLayer(network, weightMap, "model.31", std::vector<IConvolutionLayer*>{det0, det1, det2, det3}, inSize_w, inSize_h, numClass);
            yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME_YOLOV5X);
            network->markOutput(*yolo->getOutput(0));
        #endif
            // Build engine
            builder->setMaxBatchSize(maxBatchSize);
            // builder->setMaxBatchSize(8);

            config->setMaxWorkspaceSize(512 * (1 << 20));  // 16MB


        if (mode == RUN_MODE::USE_FP16)
        {
            std::cout << "Building engine WITH FP16, please wait for a while..." << std::endl;
            config->setFlag(BuilderFlag::kFP16);
        }
        else
            std::cout << "Building engine, please wait for a while..." << std::endl;

        #if defined(USE_INT8)
            std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
            assert(builder->platformHasFastInt8());
            config->setFlag(BuilderFlag::kINT8);
            Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
            config->setInt8Calibrator(calibrator);
        #endif

        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        std::cout << "Build engine successfully!" << std::endl;

        // Don't need the network any more
        network->destroy();

        // Release host memory
        for (auto& mem : weightMap)
        {
            free((void*)(mem.second.values));
        }

        return engine;
    }

    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, float& gd, float& gw, std::string& wts_name)
    {
        // Create builder
        IBuilder* builder = createInferBuilder(gLogger);
        IBuilderConfig* config = builder->createBuilderConfig();

        // Create model to populate the network, then set the outputs and create an engine
        ICudaEngine* engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
        assert(engine != nullptr);
        // Serialize the engine
        (*modelStream) = engine->serialize();

        // Close everything down
        engine->destroy();
        builder->destroy();
        config->destroy();
    }

    void doInference(float* input, float* output, int batchSize)
    {
        CUDA_CHECK(cudaSetDevice(deviceid));

        std::lock_guard<std::mutex> lk(buff_mutex);
        if (!stream)   return;

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        //fprintf(stderr, "Go in cudaMemcpyAsync copy %d bytes from %p with stream %p\n", batchSize * 3 * inSize_h * inSize_w * sizeof(float), (void*)input, (void*)stream);
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * inSize_h * inSize_w * sizeof(float), cudaMemcpyHostToDevice, stream));
        cudaStreamSynchronize(stream);

        //fprintf(stderr, "Go in execute %p\n", (void*)mTrtContext);
        mTrtContext->enqueue(batchSize, &buffers[0], stream, nullptr);
        //mTrtContext->execute(batchSize, &buffers[0]);
        cudaStreamSynchronize(stream);

        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE_YOLOV5X * sizeof(float), cudaMemcpyDeviceToHost, stream));
        //fprintf(stderr, "Go out cudaMemcpyAsync copy to %p, %d bytes\n", (void*)output,  batchSize * OUTPUT_SIZE * sizeof(float));
        cudaStreamSynchronize(stream);
    }

    void doInference(float* output, int batchSize)
    {
        CUDA_CHECK(cudaSetDevice(deviceid));

        std::lock_guard<std::mutex> lk(buff_mutex);
        if (!stream)   return;

        mTrtContext->enqueue(batchSize, &buffers[inputIndex], stream, nullptr);
        cudaStreamSynchronize(stream);

        CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE_YOLOV5X * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
    }

    void* get_cuda_stream()
    {
        return (void*)stream;
    }

    void* get_input_buff()
    {
        return buffers[inputIndex];
    }

    int getOutputSize()
    {
        return OUTPUT_SIZE_YOLOV5X;
    }

    bool is_ready() const
    {
        return net_ready;
    }

    bool net_arch_parse(std::string& netarch, float& gd, float& gw)
    {
        auto net = netarch;
        if (net == "s") {
            gd = 0.33;
            gw = 0.50;
        }
        else if (net == "m") {
            gd = 0.67;
            gw = 0.75;
        }
        else if (net == "l") {
            gd = 1.0;
            gw = 1.0;
        }
        else if (net == "x") {
            gd = 1.33;
            gw = 1.25;
        }
        else {
            return false;
        }

        return true;
    }

    void initialization(
        RUN_MODE _mode,
        int _maxBatchSize,
        int _deviceid,
        int _inSize_w,
        int _inSize_h,
        int _numClass)
    {
        mode = _mode;
        maxBatchSize = _maxBatchSize;
        deviceid = _deviceid;
        inSize_w = _inSize_w;
        inSize_h = _inSize_h;
        numClass = _numClass;
    }

    void create_engine(
        std::string& wts_name,
        std::string& engine_name)
    {
        cudaSetDevice(deviceid);

        float gd = 0.0f, gw = 0.0f;

        std::string net_arch = "x";
        if (!net_arch_parse(net_arch, gd, gw)) {
            std::cerr << "Cannot parse the CNN architecture !" << std::endl;
            exit(-1);
        }

        // create a model using the API directly and serialize it to a stream
        if (!wts_name.empty()) {
            IHostMemory* modelStream{ nullptr };
            APIToModel(maxBatchSize, &modelStream, gd, gw, wts_name);
            assert(modelStream != nullptr);
            std::ofstream p(engine_name, std::ios::binary);
            if (!p) {
                std::cerr << "could not open engine output file" << std::endl;
                exit(-1);
            }
            p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
            modelStream->destroy();

            net_ready = true;
        }
    }

    void load_engine(
        std::string& engine_name,
        int deviceid
    )
    {
        cudaSetDevice(deviceid);

        // deserialize the .engine and run inference
        std::ifstream file(engine_name, std::ios::binary);
        if (!file.good()) {
            std::cerr << "read " << engine_name << " error!" << std::endl;
            exit(-1);
        }

        char* trtModelStream = nullptr;
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();


        mTrtRunTime = createInferRuntime(gLogger);
        assert(mTrtRunTime != nullptr);
        mTrtEngine = mTrtRunTime->deserializeCudaEngine(trtModelStream, size);
        assert(mTrtEngine != nullptr);
        mTrtContext = mTrtEngine->createExecutionContext();
        assert(mTrtContext != nullptr);
        delete[] trtModelStream;
        assert(mTrtEngine->getNbBindings() == 2);

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        inputIndex = mTrtEngine->getBindingIndex(INPUT_BLOB_NAME_YOLOV5X);
        outputIndex = mTrtEngine->getBindingIndex(OUTPUT_BLOB_NAME_YOLOV5X);

        assert(inputIndex == 0);
        assert(outputIndex == 1);
        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize * 3 * inSize_h * inSize_w * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize * OUTPUT_SIZE_YOLOV5X * sizeof(float)));
        // Create stream

        CUDA_CHECK(cudaStreamCreate(&stream));

        net_ready = true;
    };
};

yolov5x_trt::yolov5x_trt()
    : impl(new implement())
{
};

yolov5x_trt::~yolov5x_trt()
{
    if (impl)
    {
        delete impl;
        impl = nullptr;
    }
}
void yolov5x_trt::initialization(
    RUN_MODE mode,
    int maxBatchSize,
    int deviceid,
    int _inSize_w,
    int _inSize_h,
    int _numClass)
{
    impl->initialization(mode, maxBatchSize, deviceid, _inSize_w, _inSize_h, _numClass);
}
void yolov5x_trt::create_engine(
    std::string& wts_name,
    std::string& engine_name)
{
    impl->create_engine(wts_name, engine_name);
}

void yolov5x_trt::load_engine(
    std::string& engine_name,
    int deviceid)
{
    impl->load_engine(engine_name, deviceid);
}

void yolov5x_trt::doInference(float* input, float* output, int batchSize)
{
    impl->doInference(input, output, batchSize);
}
void yolov5x_trt::doInference(float* output, int batchSize)
{
    impl->doInference(output, batchSize);
}
void* yolov5x_trt::get_cuda_stream()
{
    return impl->get_cuda_stream();
}
void* yolov5x_trt::get_input_buff()
{
    return impl->get_input_buff();
}
int yolov5x_trt::getOutputSize()
{
    return impl->getOutputSize();
}

bool yolov5x_trt::is_ready() const
{
    return impl->is_ready();
}