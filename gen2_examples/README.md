### Background on the DepthAI Gen2 Pipeline Builder Examples

The Gen2 Pipeline Builder (luxonis/depthai#136) allows theoretically infinite permutations of pipelines to be built. 
So there is no way to do what we did with depthai_demo.py as in Gen1 where effectively all permutations of configs can be done via CLI options.

So for Gen2 it is necessary to show how the fundamental nodes can be used in easy, and simple ways. 

This way, the pipeline consisting of multiple nodes can be easily built for a given application by piecing together these simple/clear examples.

Allowing the power of these permutations, but with the speed of copy/paste plugging together of examples.

So the below examples hope to serve to show how to use each of these nodes, so that you can piece them together for your application.

### The examples:

- [01_rgb_preview.py](https://github.com/luxonis/depthai-experiments/blob/master/gen2_examples/01_rgb_preview.py)

This example sets up a pipeline that outpus a preview of the RGB camera and displays it.

- [02_mono_preview.py](https://github.com/luxonis/depthai-experiments/blob/master/gen2_examples/02_mono_preview.py)

This example sets up a pipeline that outputs a preview of the left and right grayscale cameras and displays both.

- [03_depth_preview.py](https://github.com/luxonis/depthai-experiments/blob/master/gen2_examples/03_depth_preview.py)


- []
