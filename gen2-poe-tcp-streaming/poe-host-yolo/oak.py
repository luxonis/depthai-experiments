import depthai as dai
import time
import blobconverter

pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)

# Properties
camRgb.setPreviewSize(640, 352)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# camRgb.initialControl.setManualWhiteBalance(1)
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

def send_frame_thread(conn):
    node.warn('Sending frames..')
    try:
        while True:
            img = node.io["frame_in"].get()
            node.warn('Received frame')
            img_data = img.getData()
            ts = img.getTimestamp()

            # If there is any detection, serialize it and send it
            detections = node.io["detection_in"].get().detections

            header = "IMG " + str(ts.total_seconds()).ljust(16) + (f"{len(img_data)} {len(detections)}).ljust(16)
            node.warn(f'>{header}<')
            conn.send(bytes(header, encoding='ascii'))
            conn.send(img_data)

            for det in detections:
                txt = f"DETECTION {det.label} {(det.confidence*100):.1f} {det.xmin:.4f} {det.ymin:.4f} {det.xmax:.4f} {det.ymax:.4f}".ljust(64)
                node.warn(f'>{txt}<')
                conn.send(bytes(txt, encoding='ascii'))
    except Exception as e:
        node.warn("Client disconnected")

while True:
    conn, client = server.accept()
    node.warn(f"Connected to client IP: {client}")
    threading.Thread(target=send_frame_thread, args=(conn,)).start()
""")

with dai.Device(pipeline) as device:
    print("Connected")
    while True:
        time.sleep(1)
