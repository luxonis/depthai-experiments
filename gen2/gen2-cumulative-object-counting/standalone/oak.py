import depthai as dai
import time
import blobconverter

model = blobconverter.from_zoo(name="mobilenet-ssd", shaves=7)

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setVideoSize(640, 640)

#define a script node
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

# Define neural network
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn.setConfidenceThreshold(0.8)
nn.setBlobPath(model)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Define object tracker
objectTracker = pipeline.create(dai.node.ObjectTracker)
# objectTracker.setDetectionLabelsToTrack([0])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

#Define a video encoder
videoEnc = pipeline.create(dai.node.VideoEncoder)
videoEnc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)

# Linking
camRgb.preview.link(nn.input)
camRgb.video.link(videoEnc.input)

nn.passthrough.link(objectTracker.inputTrackerFrame)
nn.passthrough.link(objectTracker.inputDetectionFrame)
nn.out.link(objectTracker.inputDetections)

script.inputs['tracklets'].setBlocking(False)
script.inputs['tracklets'].setQueueSize(1)
objectTracker.out.link(script.inputs["tracklets"])

script.inputs['frame'].setBlocking(False)
script.inputs['frame'].setQueueSize(1)
videoEnc.bitstream.link(script.inputs['frame'])

script.setScript("""
import socket
import time
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 5000))
server.listen()
node.warn("Server up")

trackableObjects = {}
counter = [0, 0, 0, 0]  # left, right, up, down

class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]

        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False

# alternative to np.mean()
def mean(l):
    return sum(l) / len(l)

while True:
    conn, client = server.accept()
    node.warn(f"Connected to client IP: {client}")
    try:
        while True:
            pck = node.io["frame"].get()
            tracklets = node.io["tracklets"].tryGet()
            data = pck.getData()
            ts = pck.getTimestamp()

            if tracklets:
                width = 300 #pck.getWidth()
                height = 300 #pck.getHeight()
                trackletsData = tracklets.tracklets
                data_to_send = []
                for t in trackletsData:
                    to = trackableObjects.get(t.id, None)
                    # get centroid
                    roi = t.roi.denormalize(width, height)
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)
                    centroid = (int((x2-x1)/2+x1), int((y2-y1)/2+y1))

                    # If new tracklet, save its centroid
                    if t.status == Tracklet.TrackingStatus.NEW:
                        to = TrackableObject(t.id, centroid)
                    elif to is not None:
                        if not to.counted:
                            x = [c[0] for c in to.centroids]
                            direction = centroid[0] - mean(x)

                            if centroid[0] > 0.5*width and direction > 0 and mean(x) < 0.5*width:
                                counter[1] += 1
                                to.counted = True
                            elif centroid[0] < 0.5*width and direction < 0 and mean(x) > 0.5*width:
                                counter[0] += 1
                                to.counted = True

                        
                        to.centroids.append(centroid)
                    
                    if t.status != Tracklet.TrackingStatus.LOST and t.status != Tracklet.TrackingStatus.REMOVED:
                        text = "ID {}".format(t.id)

                        # prepare data for sending
                        data_to_send.append([text, centroid[0], centroid[1]])
                        
                    trackableObjects[t.id] = to
                    
            # now to send data we need to encode it (whole header is 256 characters long)
            header = f"ABCDE " + str(ts.total_seconds()).ljust(18) + str(len(data)).ljust(8) + str(counter).ljust(16) + str(data_to_send).ljust(208) 
            conn.send(bytes(header, encoding='ascii'))
            conn.send(data)
    except Exception as e:
        node.warn(f"Error oak: {e}")
        node.warn("Client disconnected")
""")

# By default, you would boot device with:
#with dai.Device(pipeline) as device:
#   while True:
#        time.sleep(1)

# But for this example, we want to flash the device with the pipeline
(f, bl) = dai.DeviceBootloader.getFirstAvailableDevice()
bootloader = dai.DeviceBootloader(bl)
progress = lambda p : print(f'Flashing progress: {p*100:.1f}%')
bootloader.flash(progress, pipeline)