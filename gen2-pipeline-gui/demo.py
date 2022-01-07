import sys
import cv2
import argparse
import depthai as dai

from pathlib import Path
from DAIPipelineGraph import DAIPipelineGraph

SCRIPT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument( "-p", "--path", type=str, default='ExampleGraph.json', help="Path to pipeline graph (Default: ExampleGraph.json)")
args = vars( parser.parse_args() )

pipeline_path = str( SCRIPT_DIR / args[ 'path' ] )

# Create the pipeline
pipeline_graph = DAIPipelineGraph( path=pipeline_path )

# Display all XLinkOut data as CV frames
with dai.Device( pipeline_graph.pipeline ) as device:
    queues = {}
    frames = {}

    for stream_id in pipeline_graph.xout_streams:
        queues[ stream_id ] = device.getOutputQueue( stream_id )
        frames[ stream_id ] = None

    while True:
        for stream_id in pipeline_graph.xout_streams:
            get_result = queues[ stream_id ].tryGet()

            # RGB Frame
            if get_result is not None:
                frames[ stream_id ] = get_result.getCvFrame()

            # SHOW IMAGE
            if frames[ stream_id ] is not None:
                cv2.imshow( stream_id, frames[ stream_id ] )

        # HANDLE QUIT
        if cv2.waitKey(1) == ord('q'):
            break