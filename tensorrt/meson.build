trtphrd_sources = [
	'gstinfpintel.cpp',
]

trtphrd_headers = [
	'gstinfpintel.h',
]


# install_headers(trtphrd_headers, subdir : inc_install_dir + '/pinference/tensorrt')
# dependencies : [gst_dep, gstvideo_dep, lib_trt_dep, opencv_dep, json_dep, glib_dep, gmodule_dep],
 
gstmypintelinf = library('gstinfpintel',
	trtphrd_sources,
	cpp_args : [cpp_args] + additional_cpp_args,
	include_directories : [configinc, '.'],
	install : true,
	install_dir : lib_install_dir,
	dependencies : [gst_dep, gstvideo_dep, lib_trt_dep, opencv_dep, json_dep, glib_dep, gmodule_dep],
)


# pkgconfig.generate(gstmypintelinf,
# 	libraries : [gst_dep],
# 	subdirs : inc_install_dir,
# 	name : 'gstmypintelinf',
# 	description : 'Tensorrt based PHRD',
# )
