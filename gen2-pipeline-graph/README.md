<img width="1201" alt="Screen Shot 2022-01-06 at 9 37 46 PM" src="https://user-images.githubusercontent.com/6503621/148497367-26ff816c-8a81-443c-9c4e-f598aa316885.png">

# Gen2 Visual Pipeline Editor
This experiment has two components:
1. A visual graph editor that lets you compose a DepthAI pipeline visually and save it as a JSON file.
2. A parser for those JSON files that will turn it into a usable DepthAI pipeline


## Editor 
The graph editor is a slightly modified version of [NodeGraphQt](https://github.com/jchanvfx/NodeGraphQt) by Johnny Chan.

### Install Dependencies:
`python3 install_requirements.py`

### Usage
`python3 pipeline_editor.py` - Runs the visual pipeline editor

### Navigation
- Press **Tab** to create new nodes
- You can find a full list of controls in the [NodeGraphQt Documentation](https://jchanvfx.github.io/NodeGraphQt/api/html/examples/ex_overview.html) 

## Parser
`DAIPipelineGraph` is the graph parser.

### Demo
`python3 demo.py` - Runs the included ExampleGraph.json pipeline

### Usage
```
from DAIPipelineGraph import DAIPipelineGraph

pipeline_graph = DAIPipelineGraph( path=pipeline_path )

with dai.Device( pipeline_graph.pipeline ) as device:
  ...
```

### Accessing Pipeline Data
- `DAIPipelineGraph.pipeline`: A reference to the DepthAI pipeline
- `DAIPipelineGraph.nodes`: A table of all the nodes. You can access them via the name you put into the "Node Name" field in the editor. Ex: `pipeline_graph.nodes["rgb_cam"].setPreviewSize(300,300)`
- `DAIPipelineGraph.xout_streams`: A list of all the names of the XLinkOut streams
