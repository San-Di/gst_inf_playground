project('playground-inf', ['c', 'cpp','cuda'],
    version : '1.0.3',
    meson_version : '>= 0.58.0',
    default_options : [ 'warning_level=1',
                      'buildtype=debugoptimized', 
					  'cpp_std=c++20'
					  ]
    )

project_name = meson.project_name()
project_version = meson.project_version()
version_arr = project_version.split('.')
version_major = version_arr[0].to_int()
version_minor = version_arr[1].to_int()
version_micro = version_arr[2].to_int()
if version_arr.length() == 4
  version_nano = version_arr[3].to_int()
else
  version_nano = 0
endif
version_is_dev = version_minor % 2 == 1 and version_micro < 90

glib_req = '>= 2.56.0'
gst_req = '>= @0@.@1@.0'.format(version_major, version_minor)
api_version = '1.0'

# Get an object returns describing a compiler
cpp = meson.get_compiler('cpp')

# Define compiler args and include directories
cpp_args = ['-DHAVE_CONFIG_H']

# Create an empty configuration object to set config.h information

plugins = []


host_system = host_machine.system()
message('Host system : ', host_system)
if host_system == 'windows'
  additional_cpp_args = ['/Zi']
elif host_system == 'linux'
  additional_cpp_args = []
endif


cdata = configuration_data()
cdata.set_quoted('PACKAGE_VERSION', project_version)
cdata.set_quoted('PACKAGE', 'custom-gstreamer-plugins')
cdata.set_quoted('GST_LICENSE', 'LGPL')
cdata.set_quoted('GST_API_VERSION', api_version)
cdata.set_quoted('GST_PACKAGE_NAME', 'GStreamer custom plugins')
cdata.set_quoted('GST_PACKAGE_ORIGIN', 'https://github.com/rosemary-crypto/Custom-Gstreamer-Plugins.git')

# Mandatory GST deps
gst_dep = dependency('gstreamer-1.0', version : gst_req,
  fallback : ['gstreamer', 'gst_dep'])
gstbase_dep = dependency('gstreamer-base-1.0', version : gst_req,
  fallback : ['gstreamer', 'gst_base_dep'])
gstnet_dep = dependency('gstreamer-net-1.0', version : gst_req,
  fallback : ['gstreamer', 'gst_net_dep'])
gstcontroller_dep = dependency('gstreamer-controller-1.0', version : gst_req,
  fallback : ['gstreamer', 'gst_controller_dep'])
gstpbutils_dep = dependency('gstreamer-pbutils-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'pbutils_dep'])
gstallocators_dep = dependency('gstreamer-allocators-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'allocators_dep'])
gstapp_dep = dependency('gstreamer-app-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'app_dep'])
gstaudio_dep = dependency('gstreamer-audio-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'audio_dep'])
gstfft_dep = dependency('gstreamer-fft-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'fft_dep'])
gstriff_dep = dependency('gstreamer-riff-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'riff_dep'])
gstrtp_dep = dependency('gstreamer-rtp-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'rtp_dep'])
gstrtsp_dep = dependency('gstreamer-rtsp-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'rtsp_dep'])
gstsdp_dep = dependency('gstreamer-sdp-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'sdp_dep'])
gsttag_dep = dependency('gstreamer-tag-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'tag_dep'])
gstvideo_dep = dependency('gstreamer-video-1.0', version : gst_req,
    fallback : ['gst-plugins-base', 'video_dep'])
gobject_dep = dependency('gobject-2.0')
glib_dep = [dependency('glib-2.0', version : glib_req, fallback: ['glib', 'libglib_dep']),
             dependency('gobject-2.0', fallback: ['glib', 'libgobject_dep'])]
gmodule_dep = dependency('gmodule-2.0', version:'>= 2.0.0', required: true)

#json dependencies
json_include_dir = join_paths('C:/vcpkg/packages/jsoncpp_x64-windows', 'include')
json_lib_dir = join_paths('C:/vcpkg/packages/jsoncpp_x64-windows', 'lib')
json_lib = cpp.find_library('jsoncpp', dirs: json_lib_dir)
json_dep = declare_dependency(
	dependencies: json_lib,
	include_directories: include_directories(json_include_dir)
)

# Dependencies for  Inference
#opencv

opencv_include_dir = 'D:/workspace/lib/opencv/4.1.2/vc140/include'
opencv_lib_dir = 'D:/workspace/lib/opencv/4.1.2/vc140/lib'
opencv_libs = ['opencv_core412', 'opencv_imgcodecs412', 'opencv_objdetect412', 'opencv_imgproc412', 'opencv_highgui412']
opencv_deps = []

foreach libname: opencv_libs
  opencv_deps += cpp.find_library(libname, dirs: opencv_lib_dir)
endforeach

opencv_dep = declare_dependency(
  dependencies: opencv_deps,
  include_directories: include_directories(opencv_include_dir)
)

#cuda
cuda_version = '11.6'

cuda_include_dir = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/include'
cuda_lib_dir = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/lib/x64'
cuda_dep = declare_dependency(
      dependencies: [cpp.find_library('cuda', dirs: cuda_lib_dir)],
      include_directories: include_directories(cuda_include_dir)
    )

cudart_dep = declare_dependency(
	dependencies: [cpp.find_library('cudart', dirs: cuda_lib_dir)],
	include_directories: include_directories(cuda_include_dir)
)

#cudnn
cudnn_version = '8.4'
cudnn_lib_dir = join_paths('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDNN/v' + cudnn_version, 'lib', 'x64')
cudnn_include_dir = join_paths('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDNN/v' + cudnn_version, 'include')
cudnn_lib = cpp.find_library('cudnn', dirs:cudnn_lib_dir, required: true)
cudnn_dep = declare_dependency(
	dependencies: cudnn_lib,
	include_directories: include_directories(cudnn_include_dir)
)

#tensorrt
trt_version = '8.4.0.6'
trt_lib_dir = join_paths('')
trt_lib_dir = join_paths('C:/Program Files/TensorRT/v' + trt_version, 'lib')
trt_include_dir = join_paths('C:/Program Files/TensorRT/v' + trt_version, 'include')
nvinfer_lib = cpp.find_library('nvinfer', dirs:trt_lib_dir, required: true)
trt_dep = declare_dependency(
		dependencies: nvinfer_lib,
		include_directories: include_directories(trt_include_dir)
)
# lib_trt_dep = [trt_dep, cuda_dep, cudart_dep, cudnn_dep]
lib_trt_dep = [trt_dep, cuda_dep, cudart_dep, cudnn_dep]
cdata.set('HAVE_TENSORRT', true)


# Define header directories
configinc = include_directories('.')


# Define installation directories
prefix = get_option('prefix')
lib_install_dir = join_paths(get_option('libdir'), 'gstreamer-1.0') #for libs
inc_install_dir = join_paths(get_option('incdir'), 'gstreamer-1.0') #for libs
plugins_install_dir = join_paths(get_option('libdir'), 'gstreamer-1.0') # for plugins
plugins_pkgconfig_install_dir = join_paths(get_option('libdir'), 'pkgconfig') #for plugins


# Import pkgconfig module
pkgconfig = import('pkgconfig')

# Install git hooks
python3 = import('python').find_installation()

# Meson will generate a header file all the entries in the configuration data object


configure_file(output : 'config.h', configuration : cdata)
run_command(python3, '-c', 'import shutil; shutil.copy("hooks/pre-commit.hook", ".git/hooks/pre-commit")')

if meson.version().version_compare('>= 0.54')
  plugin_names = []
  foreach plugin: plugins
    # FIXME: Use str.subtring() when we can depend on Meson 0.56
    split = plugin.name().split('gst')
    if split.length() == 2
      plugin_names += [split[1]]
    else
      warning('Need substring API in meson >= 0.56 to properly parse plugin name: ' + plugin.name())
      plugin_names += [plugin.name()]
    endif
  endforeach
  summary({'Plugins':plugin_names}, list_sep: ', ')
endif
subdir('temp')
