### Background on the DepthAI Gen2 Pipeline Builder Examples

The Gen2 Pipeline Builder (luxonis/depthai#136) allows theoretically infinite permutations of pipelines to be built. 
So there is no way to do what we did with depthai_demo.py as in Gen1 where effectively all permutations of configs can be done via CLI options.

So for Gen2 it is necessary to show how the fundamental nodes can be used in easy, and simple ways. 

This way, the pipeline consisting of multiple nodes can be easily built for a given application by piecing together these simple/clear examples.

Allowing the power of these permutations, but with the speed of copy/paste plugging together of examples.

So the below examples hope to serve to show how to use each of these nodes, so that you can piece them together for your application.

### The examples:

- [01_rgb_preview.py](https://github.com/luxonis/depthai-experiments/blob/master/gen2_examples/01_rgb_preview.py)

This example sets up a pipeline that outpus a preview of the RGB camera, connects over XLink to transfer these to the host real-time, and displays the RGB frames on the host with OpenCV.

- [02_mono_preview.py](https://github.com/luxonis/depthai-experiments/blob/master/gen2_examples/02_mono_preview.py)

This example sets up a pipeline that outputs the left and right grayscale camera images, connects over XLink to transfer these to the host real-time, and displays both using OpenCV.

- [03_depth_preview.py](https://github.com/luxonis/depthai-experiments/blob/master/gen2_examples/03_depth_preview.py)

This example sets up the SGBM (semi-global-matching) disparity-depth node, connects over XLink to transfer the results to the host real-time, and displays the depth map in OpenCV.

- [04_rgb_encoding.py](https://github.com/luxonis/depthai-experiments/blob/master/gen2_examples/04_rgb_encoding.py)

This example sets up the depthai video encoder in h.265 format and encodes the RGB camera input at 8MP/4K/2160p (3840x2160) at 30FPS (the maximum possible encoding resolultion possible for the encoder, higher frame-rates are possible at lower resolutions, like 1440p at 60FPS), and transfers the encoded video over XLINK to the host, saving it to disk as a video file.

Be careful with this one... it is saving video to your disk.  If you leave it running, you could fill up your storage on your host.

- [05_rgb_mono_encoding.py](https://github.com/luxonis/depthai-experiments/blob/master/gen2_examples/05_rgb_mono_encoding.py)

This example shows how to set up the encoder node to encode the RGB camera and both grayscale cameras (of DepthAI/OAK-D) at the same time.  The RGB is set to 1920x1080 and the grayscale are set to 1280x720 each, all at 30FPS.  Each encoded video stream is transferred over XLINK and saved to a respective file.

Be more careful with this one... it is saving 3 videos in parall to your disk.  If you leave it running, you could fill up your storage on your host.

- [06_rgb_full_resolution_saver.py](https://github.com/luxonis/depthai-experiments/blob/master/gen2_examples/06_rgb_full_resolution_saver.py)
- []()
- []()
- []()
- []()
- []()
- []()

This is, close to the maximum recommended combination resolution for DepthAI/OAK-D's video encoder, which is a total max pixel limit of 3840x2160 at 30FPS, which is 248,832,000 pixels/second limit.  This example runs 117,504,000 pixels/second. Above this
