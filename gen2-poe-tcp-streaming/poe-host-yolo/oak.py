import depthai as dai
import time
import blobconverter

FLASH_APP = False

pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)

# Properties
camRgb.setPreviewSize(640, 352)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setFps(40)

# Network specific settings
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setIouThreshold(0.5)
# We have this model on DepthAI Zoo, so download it using blobconverter
detectionNetwork.setBlobPath(blobconverter.from_zoo('yolov8n_coco_640x352', zoo_type='depthai'))
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)

# Create Script node that will handle TCP communication
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

# Link outputs (RGB stream, NN output) to the Script node
detectionNetwork.passthrough.link(script.inputs['frame_in'])
detectionNetwork.out.link(script.inputs['detection_in'])

script.setScript("""
import socket
import time
import threading
node.warn("Server up")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 5000))
server.listen()

while True:
    conn, client = server.accept()
    node.warn(f"Connected to client IP: {client}")

    try:
        while True:
            detections = node.io["detection_in"].get().detections
            img = node.io["frame_in"].get()
            node.warn('Received frame + dets')
            img_data = img.getData()
            ts = img.getTimestamp()

            det_arr = []
            for det in detections:
                det_arr.append(f"{det.label};{(det.confidence*100):.1f};{det.xmin:.4f};{det.ymin:.4f};{det.xmax:.4f};{det.ymax:.4f}")
            det_str = "|".join(det_arr)

            header = f"IMG {ts.total_seconds()} {len(img_data)} {len(det_str)}".ljust(32)
            node.warn(f'>{header}<')
            conn.send(bytes(header, encoding='ascii'))
            if 0 < len(det_arr):
                conn.send(bytes(det_str, encoding='ascii'))
            conn.send(img_data)
    except Exception as e:
        node.warn("Client disconnected")
""")

if FLASH_APP: # flash the app for Standalone mode
    (f, bl) = dai.DeviceBootloader.getFirstAvailableDevice()
    bootloader = dai.DeviceBootloader(bl)
    progress = lambda p : print(f'Flashing progress: {p*100:.1f}%')
    bootloader.flash(progress, pipeline)
else: # Run the app in peripheral mode
    with dai.Device(pipeline) as device:
        print("Connected")
        while True:
            time.sleep(1)
