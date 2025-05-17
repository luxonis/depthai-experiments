import time
import depthai as dai
import numpy as np
import cv2
from depthai_nodes.node import ParsingNeuralNetwork
from depthai_nodes import ImgDetectionsExtended
from utils.utils_box import reverse_resize_and_pad

from utils.img_annotation_helper import AnnotationHelper
from utils.CuboidFitter import CuboidFitter

NN_WIDTH, NN_HEIGHT = 512, 320
INPUT_SHAPE = (NN_WIDTH, NN_HEIGHT)

IMG_WIDTH, IMG_HEIGHT = 640, 400
CAMERA_RESOLUTION = (IMG_WIDTH, IMG_HEIGHT)

#device = dai.Device(dai.DeviceInfo('10.11.1.175'))
device = dai.Device()
device.setIrLaserDotProjectorIntensity(1.0)
device.setIrFloodLightIntensity(1)

platform = device.getPlatform()

# NN model init 
model_version_slug = "512x320"

model_description = dai.NNModelDescription(
    model="luxonis/yolov8-instance-segmentation-nano-carton:512x320:1.0.0",
    platform=platform.name
)

# Download or retrieve the model from the zoo
archivePath = dai.getModelFromZoo(
    model_description,
    apiKey='tapi.oUtZWL1Ib53fQxESjkwiaw.6Aa5gzkKcmhyIRpWTNtMW20Nw5cHiiqVt-BLYgi00ajRB8jy7e72VFpOehhyNR1gkt2Mn9aUtkSrOBShPuFItw'
)

nn_archive = dai.NNArchive(archivePath)

# Box measurement errors (For testing accuracy)
GROUND_TRUTH_DIMENSIONS = (24.5, 20.0, 15.5)            # Replace with real box dimensions 
total_mae = [0, 0, 0]
total_relative_error = [0, 0, 0]

def read_intrinsics():
    calibData = device.readCalibration2()
    M2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, NN_WIDTH, NN_HEIGHT))  # Because the displayed image is with NN input res 
    fx = M2[0, 0]
    fy = M2[1, 1]
    cx = M2[0, 2]
    cy = M2[1, 2]
    return fx, fy, cx, cy

class AnnotationNode(dai.node.ThreadedHostNode):
    '''
    Custom node for visualization and fitting 
    '''
    def __init__(self):
        dai.node.ThreadedHostNode.__init__(self)
        self.inputPCL = self.createInput()
        self.inputDet= self.createInput()
        self.inputRGB= self.createInput()

        self.outputPCL = self.createOutput()
        #self.outputRGB = self.createOutput()
        self.outputANN  = self.createOutput()
        self.outputANNCuboid  = self.createOutput()

        self.fitter = CuboidFitter()
        self.intrinsics = read_intrinsics()
        self.fit = False
        self.helper_det = None
        self.helper_cuboid = None

    def draw_mask(self, mask: np.ndarray, idx: int):
        """
        Trace the binary mask for a single instance and draw it as a filled polygon.
        :param mask: 2D array of class IDs
        :param idx: index of the instance to visualize
        """
        h, w = mask.shape
        binary = (mask == idx).astype(np.uint8)

        if not np.any(binary):
            return
        
        # Extract mask contours to draw polygon lines 
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        for cnt in contours:
            
            # To smooth the mask for visualization (TO DO note: could use this approach for PCL filtering)
            epsilon = 0.01 * cv2.arcLength(cnt, True)            
            approx = cv2.approxPolyDP(cnt, epsilon, True)        
            pts = approx.reshape(-1, 2)
            
            #pts = cnt.reshape(-1, 2)                   # To keep original mask 
            norm_pts = [(float(x)/w, float(y)/h) for x, y in pts]
            self.helper_det.draw_polyline(
                norm_pts,
                outline_color=None,                             # Color for mask outline
                fill_color=(1.0, 0.5, 0.5, 0.3),
                thickness=1,
                closed=True,
            )
    
    def draw_cuboid_outline(self, corners):

        fx, fy, cx_i, cy_i = self.intrinsics
        # project & normalize
        pts2d_norm = []
        for x, y, z in corners:
            if z <= 0:
                pts2d_norm.append(None)
            else:
                u = (fx * x / z + cx_i) / NN_WIDTH
                v = (fy * y / z + cy_i) / NN_HEIGHT
                pts2d_norm.append((u, v))

        # draw edges
        edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            if pts2d_norm[i] and pts2d_norm[j]:
                self.helper_cuboid.draw_line(
                    pts2d_norm[i],
                    pts2d_norm[j],
                    color=(0.0, 1.0, 0.0, 1.0),         # Green
                    thickness=3,
                )

    def fit_cuboid(self, idx: int, mask: np.ndarray, pcl: np.ndarray, pcl_color: np.ndarray,):
        """Fits cuboid and draws its 3D outline as lines."""
        
        # Prepare point-cloud 
        # Note: not tested yet (try to filetr the mask before segmenting the point cloud to remove outliers)
        """         
        binary  = (mask == idx).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        smooth_mask = np.zeros_like(binary)
        cv2.fillPoly(smooth_mask, [approx], 1)   # 1 = foreground


        pts3d = pcl.reshape((IMG_HEIGHT, IMG_WIDTH, 3))[smooth_mask==1]
        cols = cv2.cvtColor(pcl_color, cv2.COLOR_BGR2RGB)[smooth_mask==1] 
        """

        mask_bool = (mask == idx) 
        pts3d = pcl.reshape((IMG_HEIGHT, IMG_WIDTH, 3))[mask_bool]
        cols = cv2.cvtColor(pcl_color, cv2.COLOR_BGR2RGB)[mask_bool]

        self.fitter.reset()
        self.fitter.set_point_cloud(pts3d, None, cols)

        if not self.fitter.fit_orthogonal_planes():
            # Fitting failed
            self.fit = False
            return

        # Calculate dimensions and corners 
        dimensions, corners = self.fitter.calculate_dimensions_corners_MAD()

        print('Dimensions: ', dimensions)

        self.dimensions = dimensions
        self.fit = True

        corners = np.asarray(corners)
        outline = self.fitter.get_3d_lines_o3d(corners)
        corners3d = np.asarray(outline.points)
        self.draw_cuboid_outline(corners3d)

    def draw_box_and_label(self, det) -> list:
        """Draws rotated rect and label"""

        # All annotation coordinates are normalized to the NN input size (512×320)
        rr = det._rotated_rect
        cx, cy = rr.center.x, rr.center.y
        w, h = rr.size.width, rr.size.height
        angle = rr.angle

        self.helper_det.draw_rotated_rect(
            (cx, cy), 
            (w, h),
            angle,
            fill_color=None,
            thickness=2,
        )

        # choose first corner for label
        corners = rr.getPoints()
        corner0 = (corners[0].x, corners[0].y)

        if self.fit:
            label = (
            f"Box ({det._confidence:.2f}) "
            f"{self.dimensions[0]:.1f} x {self.dimensions[1]:.1f} x {self.dimensions[2]:.1f} cm")

        else:
            label = f"{'Box'} {det._confidence:.2f}"

        self.helper_det.draw_text(
            label,
            corner0,
            size=18,
        )
    
    def annotate_detection(self, det, idx: int, mask: np.ndarray, pcl, pcl_colors):
        """Draw all annotations (mask, 3D box fit, bounding box + label) for a single detection."""
        # Draw segmentation mask 
        self.draw_mask(mask, idx)

        # Cuboid fitting 
        self.fit_cuboid(idx, mask, pcl, pcl_colors)

        # Draw bbox and label 
        self.draw_box_and_label(det)

    def run(self):

        while self.isRunning():

            try:
                inPointCloud = self.inputPCL.get()
                inRGB = self.inputRGB.get()
                parser_output: ImgDetectionsExtended = self.inputDet.get()
            except dai.MessageQueue.QueueException:
                return # Pipeline closed

            points, colors = inPointCloud.getPointsRGB()
            rgba_img = colors.reshape(IMG_HEIGHT, IMG_WIDTH, 4)
            bgr_img = cv2.cvtColor(rgba_img, cv2.COLOR_BGRA2BGR)        # For PCL color 
                
            frame = inRGB.getCvFrame()
            timestamp = inRGB.getTimestamp()
            seq_num = inRGB.getSequenceNum()

            mask = parser_output._masks._mask
            detections = parser_output.detections
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mask_full = reverse_resize_and_pad(mask, (IMG_WIDTH, IMG_HEIGHT), INPUT_SHAPE)

            self.helper_det = AnnotationHelper()
            self.helper_cuboid = AnnotationHelper()                 # To display cuboid fitting result outline in another topic

            for idx, det in enumerate(detections):

                self.annotate_detection(det, idx, mask_full, points, bgr_img)

            # send annotations
            ann_msg = self.helper_det.build(timestamp, seq_num)
            ann_msg_cuboid = self.helper_cuboid.build(timestamp, seq_num)

            self.outputANN.send(ann_msg)
            self.outputANNCuboid.send(ann_msg_cuboid)
            #self.outputRGB.send(inRGB)

# Create pipeline

with dai.Pipeline(device) as p:

    fps = 20

    color = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    color_output = color.requestOutput(
        CAMERA_RESOLUTION, dai.ImgFrame.Type.RGB888i, fps=fps
    )

    left = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = p.create(dai.node.StereoDepth).build(
        left=left.requestOutput(CAMERA_RESOLUTION, fps=fps),
        right=right.requestOutput(CAMERA_RESOLUTION, fps=fps),
    )
    stereo.initialConfig.setMedianFilter(dai.StereoDepthConfig.MedianFilter.KERNEL_7x7)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.enableDistortionCorrection(True)
    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)

    align = p.create(dai.node.ImageAlign)
    stereo.depth.link(align.input)
    color_output.link(align.inputAlignTo)

    # For PCL
    rgbd = p.create(dai.node.RGBD).build()
    align.outputAligned.link(rgbd.inDepth)
    color_output.link(rgbd.inColor)

    # For NN 
    manip = p.create(dai.node.ImageManipV2)
    manip.initialConfig.setOutputSize(*nn_archive.getInputSize())
    manip.initialConfig.setFrameType(
        dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
    )
    manip.setMaxOutputFrameSize(
        nn_archive.getInputSize()[0] * nn_archive.getInputSize()[1] * 3
    )

    color_output.link(manip.inputImage)

    nn = p.create(ParsingNeuralNetwork).build(
        nn_source=nn_archive, input=manip.out
    )

    nn._parsers[0].setConfidenceThreshold(0.7)
    nn._parsers[0].setIouThreshold(0.5)
    nn._parsers[0].setMaskConfidence(0.5)

    Annotations = AnnotationNode()

    rgbd.pcl.link(Annotations.inputPCL)
    nn.passthrough.link(Annotations.inputRGB)
    nn.out.link(Annotations.inputDet)

    outputToVisualize = color.requestOutput(
        (1280, 800),                                       # Needs to have same aspect ratio as NN input (1.6) for it to work
        type=dai.ImgFrame.Type.NV12
    )

    vis = dai.RemoteConnection(httpPort=8082)

    vis.addTopic("Raw video",  outputToVisualize, "images")
    #vis.addTopic("Video", Annotations.outputRGB, "images")
    vis.addTopic("AnnotationsYOLO", Annotations.outputANN, "img_annotations")
    vis.addTopic("AnnotationsCuboidFit", Annotations.outputANNCuboid, "img_annotations")
    #vis.addTopic("Detections", Annotations.outputPCL, "pointcloud")

    p.start()
    vis.registerPipeline(p)
    while p.isRunning():
        time.sleep(0.1)
