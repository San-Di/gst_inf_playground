
#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <gst/gst.h>
#include <iostream>
#include "gstinfpintel.h"

GST_DEBUG_CATEGORY_STATIC (gst_inf_pinteldebug);
#define GST_CAT_DEFAULT gst_inf_pinteldebug

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT
};

/* the capabilities of the inputs and outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("ANY")
    );

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("ANY")
    );

#define gst_inf_pintelparent_class parent_class
G_DEFINE_TYPE(GstInfPintel, gst_inf_pintel, GST_TYPE_ELEMENT);

static void gst_inf_pintelset_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_inf_pintelget_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_inf_pintel_sink_event (GstPad * pad, GstObject * parent, GstEvent * event);
static GstFlowReturn gst_inf_pintel_chain (GstPad * pad, GstObject * parent, GstBuffer * buf);
static GstFlowReturn gst_video_inference_process_model(GstInfPintel * self, GstBuffer * buffer, GstPad* pad);
static gboolean gst_video_inference_model_run_prediction (GstInfPintel * self, GstPad * pad, GstBuffer * buffer, gpointer * prediction_data, gsize * prediction_size);

/* ========= Preprocess functions ===============*/
static void prepareImage(cv::Mat& img, GstInfPintel *self);
static void do_yolo(GstInference * self);
//===============================================


static void do_yolo(GstInference * self)
{		
	// const int one_batch_data_size = m_config->neth * m_config->netw * 3 ;
  g_print("============== During YOLO ===============");

	const int one_batch_data_size = neth * netw * 3;
	while (true)
	{

		if (self->m_data_depot.empty()) continue;

		int batch_idx = MIN(self->m_data_depot.size(), max_batch) - 1; //batch_idx is suppossed to be (batch-1)
		
		if (batch_idx < 0 || batch_idx >= max_batch) continue; //batch_idx is suppossed to be (batch-1)

		std::vector<yolo_data> temp_data_depot;
		for (int dataIdx = 0; dataIdx <= batch_idx; ++dataIdx)
		{
			auto item = self->m_data_depot.front();
			self->m_data_depot.pop_front();

			temp_data_depot.push_back(item);
		}

		self->net_input_buff = (float*)self->m_yolo->get_input();
		
		std::vector<cv::Size> img_size_vec;		
		std::vector<std::vector<bbox_t>> result_vec;
		if (self->net_input_buff)
		{
			for (int dataIdx = 0; dataIdx < temp_data_depot.size(); ++dataIdx)
			{
				CUDA_CHECK(cudaMemcpyAsync(self->net_input_buff, temp_data_depot.at(dataIdx).feed_data, one_batch_data_size * sizeof(float), cudaMemcpyDeviceToDevice, self->stream));
				CUDA_CHECK(cudaStreamSynchronize(self->stream));

				self->net_input_buff += one_batch_data_size;
				img_size_vec.push_back(temp_data_depot.at(dataIdx).im_size);
			}
			
			//do inference
			self->m_yolo->inference(img_size_vec.size());
			self->m_yolo->get_output(img_size_vec, result_vec, temp_data_depot.at(0)._threshold);
		}

		//notify the result
		for (int resultIdx = 0; resultIdx < temp_data_depot.size() && resultIdx < result_vec.size(); ++resultIdx)
		{			
			// std::lock_guard<std::mutex> locker(m_result_mutex[temp_data_depot.at(resultIdx)._channel]);
			
			self->m_result_repot[temp_data_depot.at(resultIdx)._channel].swap(result_vec.at(resultIdx));
			self->m_finish_forward[temp_data_depot.at(resultIdx)._channel] = true;
		}
		// m_result_cv.notify_all();
		std::cout<<"Done"<<std::endl;

	}
}

static void prepareImage(cv::Mat& img, GstInfPintel *self)
{

  g_print("============== During PrepareImage ===============");
	int c = 3;
	int h = 320;   //net h
	int w = 320;   //net w

	cv::Mat cropped(h, w, CV_8UC3, 127);
	cv::resize(img, cropped, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
	cv::cvtColor(cropped, cropped, cv::COLOR_BGR2RGB);

	cv::Mat img_float;
	if (c == 3)
		cropped.convertTo(img_float, CV_32FC3, 1 / 255.0);
	else
		cropped.convertTo(img_float, CV_32FC1, 1 / 255.0);

	//HWC TO CHW
	std::vector<cv::Mat> input_channels(c);
	cv::split(img_float, input_channels);

	int channelLength = h * w;
	float* data;
	for (int i = 0; i < c; ++i) {
		memcpy(data, input_channels.at(i).data, channelLength * sizeof(float));
		data += channelLength;
	}

   	cv::Size cv_size = cv::Size(w,h);

	yolo_data tem_data = { cv_size , data , 3 , 0.2 };
	self->m_data_depot.push_back(tem_data);
}


static GstFlowReturn gst_video_inference_process_model(GstInfPintel * self, GstBuffer * buffer, GstPad* pad)
{

	GstFlowReturn ret = GST_FLOW_OK;
	GstMeta *current_meta = NULL;
	GstMeta *meta_model = NULL;
	GstVideoInfo *info_model = NULL;
	GstBuffer *buffer_model = NULL;
	gpointer prediction_data = NULL;
	gsize prediction_size;
	gboolean pred_valid = FALSE;

	g_return_val_if_fail(self != NULL, GST_FLOW_ERROR);
	g_return_val_if_fail(buffer != NULL, GST_FLOW_ERROR);
	g_return_val_if_fail(pad != NULL, GST_FLOW_ERROR);

	GST_LOG_OBJECT(self, "Processing model buffer");

	buffer_model = gst_buffer_make_writable(buffer);
	current_meta = gst_buffer_get_meta(buffer_model, gst_inference_meta_api_get_type ());
	
  gst_video_inference_model_run_prediction(self, pad, buffer_model, &prediction_data, &prediction_size);

	/* Assign already created inferencemeta, no need to create a new one */
	meta_model = current_meta;
	return GST_FLOW_OK;
}

static gboolean gst_video_inference_model_run_prediction (GstInfPintel * self, GstPad * pad, GstBuffer * buffer, gpointer * prediction_data, gsize * prediction_size)
{
	GstVideoFrame *inframe, *outframe;
	GstBuffer *outbuf;
	gboolean ret;
	guint width, height, channels;
	const guint num_dims = 2;
	gint sizes[2] = { 0, 0 };
  	gsize steps[2] = { 0, 0 };
	cv::Mat cv_mat;
	
	gchar *data = NULL;

	g_return_val_if_fail(self, FALSE);
	// g_return_val_if_fail(klass, FALSE);
	g_return_val_if_fail(pad, FALSE);
	g_return_val_if_fail(buffer, FALSE);
	g_return_val_if_fail(prediction_data, FALSE);
	g_return_val_if_fail(prediction_size, FALSE);

	video_inference_map_buffers(pad, buffer, inframe, outframe);
	outbuf = outframe->buffer;

	channels = GST_VIDEO_FRAME_COMP_PSTRIDE (inframe, 0);
	width = GST_VIDEO_FRAME_WIDTH (inframe);
	height = GST_VIDEO_FRAME_HEIGHT (inframe);
	data = (gchar *) GST_VIDEO_FRAME_PLANE_DATA (inframe, 0);

	sizes[0] = height;
	sizes[1] = width;

	/* This is not a mistake, it's oddly inverted */
	steps[1] = channels;
	steps[0] = GST_VIDEO_FRAME_COMP_STRIDE (inframe, 0);


	cv_mat = cv::Mat(num_dims, (gint *) sizes, CV_MAKETYPE (CV_8U, channels), data, (gsize *) steps);

  g_print("============== Before PrepareImage ===============");

	prepareImage(cv_mat, self);

  g_print("============== Before YOLO ===============");
	do_yolo(self);

	gst_video_frame_unmap(inframe);
	gst_video_frame_unmap(outframe);
	gst_buffer_unref(outbuf);

	return true;
}

static void
gst_inf_pintel_class_init (GstInfPintelClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_inf_pintelset_property;
  gobject_class->get_property = gst_inf_pintelget_property;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gst_element_class_set_details_simple(gstelement_class,
    "MyFilter",
    "FIXME:Generic",
    "FIXME:Generic Template Element",
    "Ozan Karaali <<user@hostname.org>>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));

}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_inf_pintel_init (GstInfPintel * filter)
{
  filter->sinkpad = gst_pad_new_from_static_template (&sink_factory, "sink");
  gst_pad_set_event_function (filter->sinkpad,
                              GST_DEBUG_FUNCPTR(gst_inf_pintel_sink_event));
  gst_pad_set_chain_function (filter->sinkpad,
                              GST_DEBUG_FUNCPTR(gst_inf_pintel_chain));
  GST_PAD_SET_PROXY_CAPS (filter->sinkpad);
  gst_element_add_pad (GST_ELEMENT (filter), filter->sinkpad);

  filter->srcpad = gst_pad_new_from_static_template (&src_factory, "src");
  GST_PAD_SET_PROXY_CAPS (filter->srcpad);
  gst_element_add_pad (GST_ELEMENT (filter), filter->srcpad);

  filter->silent = FALSE;
  filter->api = std::make_shared<pintel_backend_api>();
  filter->api->load_engine();
}

static void
gst_inf_pintelset_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstInfPintel *filter = GST_INF_PINTEL (object);

  switch (prop_id) {
    case PROP_SILENT:
      filter->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_inf_pintelget_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstInfPintel *filter = GST_INF_PINTEL (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, filter->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* GstElement vmethod implementations */

/* this function handles sink events */
static gboolean
gst_inf_pintel_sink_event (GstPad * pad, GstObject * parent, GstEvent * event)
{
  GstInfPintel *filter;
  gboolean ret;

  filter = GST_INF_PINTEL (parent);

  GST_LOG_OBJECT (filter, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps * caps;

      gst_event_parse_caps (event, &caps);
      /* do something with the caps */

      /* and forward */
      ret = gst_pad_event_default (pad, parent, event);
      break;
    }
    default:
      ret = gst_pad_event_default (pad, parent, event);
      break;
  }
  return ret;
}

/* chain function
 * this function does the actual processing
 */
static GstFlowReturn
gst_inf_pintel_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstInfPintel *filter;

  filter = GST_INF_PINTEL (parent);

  if (filter->silent == FALSE){
    g_print ("Loaded!");
    // Now we can use iostream C++:
    std::cout<< "Test" <<std::endl;
    g_print("INFERENCE : \n\twidth = %d, format  = %s, start = %d, stop  = %d\n",width, format, start, stop);

    inference = GST_INFERENCE (parent);

    g_return_val_if_fail(pad != NULL, GST_FLOW_ERROR);
    g_return_val_if_fail(inference != NULL, GST_FLOW_ERROR);

    GstFlowReturn ret = gst_video_inference_process_model(inference, buffer, pad);
  }
  return gst_pad_push (filter->srcpad, buf);
// test_yolo(0);
}

/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
infpintel_init (GstPlugin * infpintel)
{
  /* debug category for fltering log messages
   *
   * exchange the string 'Template infpintel' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_inf_pinteldebug, "infpintel",
      0, "Template infpintel");

  return gst_element_register (infpintel, "infpintel", GST_RANK_NONE,
      GST_TYPE_INF_PINTEL);
}

/* PACKAGE: this is usually set by autotools depending on some _INIT macro
 * in configure.ac and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use autotools to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "myfirstinfpintel"
#endif

/* gstreamer looks for this structure to register infpintels
 *
 * exchange the string 'Template infpintel' with your infpintel description
 */
GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    infpintel,
    "Template infpintel",
    infpintel_init,
    PACKAGE_VERSION,
    "LGPL",
    "GStreamer",
    "http://https://github.com/rosemary-crypto/Custom-Gstreamer-Plugins/test_yolo"
)
