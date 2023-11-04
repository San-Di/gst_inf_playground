meson setup --wipe build --prefix C:\gstreamer_plugins --libdir C:\gstreamer_plugins
gst-inspect-1.0.exe infpintel
gst-launch-1.0 audiotestsrc num-buffers=10 ! infpintel ! fakesink sync=false