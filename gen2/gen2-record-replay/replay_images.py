from depthai_sdk import Replay
import depthai  as dai
import numpy as np
import cv2

# Frames are already rectified, so we don't need calib.json
replay = Replay("replay_data/stereo_frames")
        
pipeline, nodes = replay.initPipeline()

nodes.stereo.setLeftRightCheck(True)
nodes.stereo.setSubpixel(True)
nodes.stereo.setExtendedDisparity(False)

xoutNN = pipeline.create(dai.node.XLinkOut)
xoutNN.setStreamName("disparity")
nodes.stereo.disparity.link(xoutNN.input)

with dai.Device(pipeline) as device:
    replay.createQueues(device)
    depthQ = device.getOutputQueue("disparity")
    multiplier = 255 / nodes.stereo.initialConfig.getMaxDisparity()
    while replay.sendFrames():
        imgFrame = depthQ.get()
        disparity = (imgFrame.getCvFrame() * multiplier).astype(np.uint8)
        disparity = cv2.putText(disparity, str(imgFrame.getSequenceNum()), (20, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
        cv2.imshow("Stereo disparity", disparity)

        if cv2.waitKey(1) == ord('q'):
            break
