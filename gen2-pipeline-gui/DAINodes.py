import sys
import re

from Qt import QtWidgets
from NodeGraphQt import NodeGraph, BaseNode, setup_context_menu, NodeBaseWidget
from DAINodeWidgets import *

board_sockets = [ 'AUTO', 'RGB', 'LEFT', 'RIGHT' ]
orientations = [ 'AUTO', 'NORMAL', 'HORIZONTAL_MIRROR', 'VERTICAL_FLIP', 'ROTATE_180_DEG' ]
color_resolutions = [ 'THE_1080_P', 'THE_4_K', 'THE_12_MP', 'THE_13_MP' ]
mono_resolutions = [ 'THE_720_P', 'THE_800_P', 'THE_480_P', 'THE_400_P' ]
tracker_types = [ 'SHORT_TERM_KCF', 'SHORT_TERM_IMAGELESS', 'ZERO_TERM_COLOR_HISTOGRAM', 'ZERO_TERM_IMAGELESS' ]
tracker_id_policy = [ 'SMALLEST_ID', 'UNIQUE_ID' ]
encoder_profiles = [ 'H264_BASELINE', 'H264_HIGH', 'H264_MAIN', 'H265_MAIN', 'MJPEG' ]
median_filters = [ 'MEDIAN_OFF', 'KERNEL_3x3', 'KERNEL_5x5', 'KERNEL_7x7' ] 


class DAINodeGraph(NodeGraph):

    def __init__(self):
        super(DAINodeGraph, self).__init__()

        self.register_node(XLinkInNode)
        self.register_node(XLinkOutNode)
        self.register_node(ColorCameraNode)
        self.register_node(EdgeDetectorNode)
        self.register_node(FeatureTrackerNode)
        self.register_node(ImageManipNode)
        self.register_node(IMUNode)
        self.register_node(MobileNetDetectionNetworkNode)
        self.register_node(MobileNetSpatialDetectionNetworkNode)
        self.register_node(MonoCameraNode)
        self.register_node(NeuralNetworkNode)
        self.register_node(ObjectTrackerNode)
        self.register_node(ScriptNode)
        self.register_node(SPIInNode)
        self.register_node(SPIOutNode)
        self.register_node(StereoDepthNode)
        self.register_node(SystemLoggerNode)
        self.register_node(VideoEncoderNode)
        self.register_node(YoloDetectionNetworkNode)
        self.register_node(YoloSpatialDetectionNetworkNode)
    

class DepthAINode(BaseNode):
    # unique node identifier domain.
    __identifier__ = 'depthai'

    node_count = {}

    def __init__(self):
        super(DepthAINode, self).__init__()

        if not self.NODE_NAME in DepthAINode.node_count:
            DepthAINode.node_count[ self.NODE_NAME ] = 0

        count = DepthAINode.node_count[ self.NODE_NAME ]
        DepthAINode.node_count[ self.NODE_NAME ] = count + 1

        use_label = 'Node Name'
        use_name = self.NODE_NAME + str( count )

        stream_nodes = [ XLinkInNode, XLinkOutNode, SPIInNode, SPIOutNode ]
        if type( self ) in stream_nodes:
            use_label = 'Node/Stream Name'

        self.add_text_input( 'node_name', use_label, use_name, no_spaces=True )


class XLinkInNode(DepthAINode):

    # initial default node name.
    NODE_NAME = 'XLinkIn'

    def __init__(self):
        super(XLinkInNode, self).__init__()

        self.add_output('out')


class XLinkOutNode(DepthAINode):

    NODE_NAME = 'XLinkOut'

    def __init__(self):
        super(XLinkOutNode, self).__init__()
        self.add_input('input')

        self.add_float_input( 'fps_limit', 'FPS Limit', -1.0 )
        self.add_checkbox( 'get_metadata_only', 'Get Metadata Only', state=False ) 
        

class ColorCameraNode(DepthAINode):

    NODE_NAME = 'ColorCamera'

    def __init__(self):
        super(ColorCameraNode, self).__init__()
        
        self.add_input('inputConfig', data_type='ImageManipConfig')
        self.add_input('inputControl', data_type='CameraControl')

        self.add_output('raw', data_type='ImgFrame')
        self.add_output('isp', data_type='ImgFrame')
        self.add_output('still', data_type='ImgFrame')
        self.add_output('preview', data_type='ImgFrame')
        self.add_output('video', data_type='ImgFrame')

        self.add_combo_menu('board_socket', 'Board Socket', board_sockets)
        self.add_combo_menu('orientation', 'Orientation', orientations)
        self.add_combo_menu('resolution', 'Resolution', color_resolutions)
        self.add_checkbox( 'interleaved', 'Interleaved', state=False ) 


class EdgeDetectorNode(DepthAINode):

    NODE_NAME = 'EdgeDetector'

    def __init__(self):
        super(EdgeDetectorNode, self).__init__()
        
        self.add_input('inputConfig', data_type='EdgeDetectorConfig')
        self.add_input('inputImage', data_type='ImgFrame')
        self.add_output('outputImage', data_type='ImgFrame')

        self.add_checkbox( 'wait_for_config_input', 'Wait For Config Input', state=False ) 


class FeatureTrackerNode(DepthAINode):

    NODE_NAME = 'FeatureTracker'

    def __init__(self):
        super(FeatureTrackerNode, self).__init__()
        
        self.add_input('inputConfig', data_type='FeatureTrackerConfig')
        self.add_input('inputImage', data_type='ImgFrame')
        self.add_output('outputFeatures', data_type='TrackedFeatures')
        self.add_output('passthroughInputImage', data_type='ImgFrame')
        
        self.add_int_input( 'num_shaves', 'NumShaves', '2' )
        self.add_int_input( 'num_memory_slices', 'NumMemorySlices', '2' )
        self.add_checkbox( 'wait_for_config_input', 'Wait For Config Input', state=False ) 


class ImageManipNode(DepthAINode):

    NODE_NAME = 'ImageManip'

    def __init__(self):
        super(ImageManipNode, self).__init__()
        
        self.add_input('inputImage', data_type='ImgFrame')
        self.add_input('inputConfig', data_type='ImageManipConfig')
        self.add_output('out', data_type='ImgFrame')


class IMUNode(DepthAINode):

    NODE_NAME = 'IMU'

    def __init__(self):
        super(IMUNode, self).__init__()
        
        self.add_output('out', data_type='ImgFrame')

        self.add_int_input( 'report_rate', 'Report Rate', '100' )
        self.add_int_input( 'batch_report_threshold', 'Batch Report Threshold', '1' )
        self.add_int_input( 'max_batch_reports', 'Max Batch Reports', '10' )

        self.add_checkbox( 'ACCELEROMETER_RAW', 'ACCELEROMETER_RAW', state=False ) 
        self.add_checkbox( 'ACCELEROMETER', 'ACCELEROMETER', state=False ) 
        self.add_checkbox( 'LINEAR_ACCELERATION', 'LINEAR_ACCELERATION', state=False ) 
        self.add_checkbox( 'GRAVITY', 'GRAVITY', state=False ) 
        self.add_checkbox( 'GYROSCOPE_RAW', 'GYROSCOPE_RAW', state=False ) 
        self.add_checkbox( 'GYROSCOPE_CALIBRATED', 'GYROSCOPE_CALIBRATED', state=False ) 
        self.add_checkbox( 'GYROSCOPE_UNCALIBRATED', 'GYROSCOPE_UNCALIBRATED', state=False ) 
        self.add_checkbox( 'MAGNETOMETER_RAW', 'MAGNETOMETER_RAW', state=False ) 
        self.add_checkbox( 'MAGNETOMETER_CALIBRATED', 'MAGNETOMETER_CALIBRATED', state=False ) 
        self.add_checkbox( 'MAGNETOMETER_UNCALIBRATED', 'MAGNETOMETER_UNCALIBRATED', state=False ) 
        self.add_checkbox( 'ROTATION_VECTOR', 'ROTATION_VECTOR', state=False ) 
        self.add_checkbox( 'GAME_ROTATION_VECTOR', 'GAME_ROTATION_VECTOR', state=False ) 
        self.add_checkbox( 'GEOMAGNETIC_ROTATION_VECTOR', 'GEOMAGNETIC_ROTATION_VECTOR', state=False ) 
        self.add_checkbox( 'ARVR_STABILIZED_ROTATION_VECTOR', 'ARVR_STABILIZED_ROTATION_VECTOR', state=False ) 
        self.add_checkbox( 'ARVR_STABILIZED_GAME_ROTATION_VECTOR', 'ARVR_STABILIZED_GAME_ROTATION_VECTOR', state=False )


class MobileNetDetectionNetworkNode(DepthAINode):

    NODE_NAME = 'MobileNetDetectionNetwork'

    def __init__(self):
        super(MobileNetDetectionNetworkNode, self).__init__()
        
        self.add_input('input')
        self.add_output('out', data_type='ImgDetections')
        self.add_output('passthrough', data_type='ImgFrame')
        self.add_float_input( 'confidence_threshold', 'Confidence Threshold', 0.5 )


class MobileNetSpatialDetectionNetworkNode(DepthAINode):

    NODE_NAME = 'MobileNetSpatialDetectionNetwork'

    def __init__(self):
        super(MobileNetSpatialDetectionNetworkNode, self).__init__()
        
        self.add_input('input', data_type='ImgFrame')
        self.add_input('inputDepth', data_type='ImgFrame')

        self.add_output('passthrough', data_type='ImgFrame')
        self.add_output('out', data_type='SpatialImgDetections')
        self.add_output('boundingBoxMapping', data_type='SpatialLocationCalculatorConfig')
        self.add_output('passthroughDepth', data_type='ImgFrame')

        self.add_float_input( 'confidence_threshold', 'Confidence Threshold', 0.5 )
        self.add_float_input( 'bounding_box_scale', 'Bounding Box Scale Factor', 0.5 )
        self.add_float_input( 'lower_depth_threshold', 'Lower Depth Threshold (mm)', 100 )
        self.add_float_input( 'upper_depth_threshold', 'Upper Depth Threshold (mm)', 5000 )


class MonoCameraNode(DepthAINode):

    NODE_NAME = 'MonoCamera'

    def __init__(self):
        super(MonoCameraNode, self).__init__()
        
        self.add_input('inputControl', data_type='CameraControl')

        self.add_output('out', data_type='ImgFrame')

        self.add_combo_menu('board_socket', 'Board Socket', board_sockets)
        self.add_combo_menu('orientation', 'Orientation', orientations)
        self.add_combo_menu('resolution', 'Resolution', mono_resolutions)


class NeuralNetworkNode(DepthAINode):

    NODE_NAME = 'NeuralNetwork'

    def __init__(self):
        super(NeuralNetworkNode, self).__init__()
        
        self.add_input('input')
        self.add_output('out', data_type='NNData')
        self.add_output('passthrough', data_type='ImgFrame')


class ObjectTrackerNode(DepthAINode):

    NODE_NAME = 'ObjectTracker'

    def __init__(self):
        super(ObjectTrackerNode, self).__init__()
        
        self.add_input('inputDetectionFrame', data_type='ImgFrame')
        self.add_input('inputTrackerFrame', data_type='ImgFrame')
        self.add_input('inputDetections', data_type='ImgDetections')

        self.add_output('out', data_type='Tracklets')
        self.add_output('passthroughDetectionFrame', data_type='ImgFrame')
        self.add_output('passthroughTrackerFrame', data_type='ImgFrame')
        self.add_output('passthroughDetections', data_type='ImgDetections')

        self.add_combo_menu('type', 'Tracker Type', tracker_types)
        self.add_combo_menu('id_policy', 'ID Policy', tracker_id_policy)
        self.add_float_input( 'threshold', 'Tracker Threshold', 0 )


class ScriptNode(DepthAINode):

    NODE_NAME = 'Script'

    def __init__(self):
        super(ScriptNode, self).__init__()
        
        port_widget = AddPortWidgetWrapper(self.view)
        self.add_custom_widget(port_widget, tab='Custom')


class SpatialLocationCalculatorNode(DepthAINode):

    NODE_NAME = 'SpatialLocationCalculator'

    def __init__(self):
        super(SpatialLocationCalculatorNode, self).__init__()
        
        self.add_input('inputConfig', data_type='SpatialLocationCalculatorConfig')
        self.add_input('inputDepth', data_type='ImgFrame')

        self.add_output('out', data_type='SpatialLocationCalculatorData')
        self.add_output('passthroughDepth', data_type='ImgFrame')

        self.add_checkbox( 'wait_for_config_input', 'Wait For Config Input', state=False ) 


class SPIInNode(DepthAINode):

    # initial default node name.
    NODE_NAME = 'SPIIn'

    def __init__(self):
        super(SPIInNode, self).__init__()

        self.add_output('out')
        self.add_int_input( 'bus_id', 'Bus ID', '0' )


class SPIOutNode(DepthAINode):

    # initial default node name.
    NODE_NAME = 'SPIOut'

    def __init__(self):
        super(SPIOutNode, self).__init__()

        self.add_input('input')
        self.add_int_input( 'bus_id', 'Bus ID', '0' )


class StereoDepthNode(DepthAINode):

    # initial default node name.
    NODE_NAME = 'StereoDepth'

    def __init__(self):
        super(StereoDepthNode, self).__init__()

        self.add_input('left', data_type='ImgFrame')
        self.add_input('right', data_type='ImgFrame')
        self.add_input('inputConfig', data_type='StereoDepthConfig')

        self.add_output('confidenceMap', data_type='ImgFrame')
        self.add_output('rectifiedLeft', data_type='ImgFrame')
        self.add_output('syncedLeft', data_type='ImgFrame')
        self.add_output('depth', data_type='ImgFrame')
        self.add_output('disparity', data_type='ImgFrame')
        self.add_output('rectifiedRight', data_type='ImgFrame')
        self.add_output('syncedRight', data_type='ImgFrame')
        self.add_output('outConfig', data_type='StereoDepthConfig')
        
        self.add_checkbox( 'lr_check', 'Left Right Check', state=False ) 
        self.add_checkbox( 'extended_disparity', 'Extended Disparity', state=False ) 
        self.add_checkbox( 'subpixel', 'Subpixel', state=False ) 
        self.add_combo_menu('median_filter', 'Median Filter', median_filters )
        self.add_int_input( 'confidence_threshold', 'Confidence Threshold', '200' )


class SystemLoggerNode(DepthAINode):

    # initial default node name.
    NODE_NAME = 'SystemLogger'

    def __init__(self):
        super(SystemLoggerNode, self).__init__()

        self.add_output('out', data_type='SystemInformation')
        self.add_float_input( 'rate', 'Logging Rate (hz)', '1' )


class VideoEncoderNode(DepthAINode):

    # initial default node name.
    NODE_NAME = 'VideoEncoder'

    def __init__(self):
        super(VideoEncoderNode, self).__init__()

        self.add_input('input', data_type='ImgFrame')
        self.add_output('bitstream', data_type='ImgFrame')
        self.add_int_input( 'quality', 'Quality', '100', range=(0,100) )


class YoloDetectionNetworkNode(DepthAINode):

    # initial default node name.
    NODE_NAME = 'YoloDetectionNetwork'

    def __init__(self):
        super(YoloDetectionNetworkNode, self).__init__()

        self.add_input('input', data_type='ImgFrame')
        self.add_output('out', data_type='ImgDetections')
        self.add_output('passthrough', data_type='ImgFrame')
        
        self.add_float_input( 'confidence_threshold', 'Confidence Threshold', '0.5' )
        self.add_float_input( 'iou_threshold', 'IOU Threshold', '0.5' )
        self.add_int_input( 'coordinate_size', 'Coordinate Size', '4' )


class YoloSpatialDetectionNetworkNode(DepthAINode):

    # initial default node name.
    NODE_NAME = 'YoloSpatialDetectionNetwork'

    def __init__(self):
        super(YoloSpatialDetectionNetworkNode, self).__init__()

        self.add_input('input', data_type='ImgFrame')
        self.add_input('inputDepth', data_type='ImgFrame')

        self.add_output('passthrough', data_type='ImgFrame')
        self.add_output('out', data_type='SpatialImgDetections')
        self.add_output('boundingBoxMapping', data_type='SpatialLocationCalculatorConfig')
        self.add_output('passthroughDepth', data_type='ImgFrame')
        
        self.add_float_input( 'confidence_threshold', 'Confidence Threshold', '0.5' )
        self.add_float_input( 'iou_threshold', 'IOU Threshold', '0.5' )
        self.add_float_input( 'bounding_box_scale', 'Bounding Box Scale Factor', 0.5 )
        self.add_float_input( 'lower_depth_threshold', 'Lower Depth Threshold (mm)', 100 )
        self.add_float_input( 'upper_depth_threshold', 'Upper Depth Threshold (mm)', 5000 )
        self.add_int_input( 'coordinate_size', 'Coordinate Size', '4' )

