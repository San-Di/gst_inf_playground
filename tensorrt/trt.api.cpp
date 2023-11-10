
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