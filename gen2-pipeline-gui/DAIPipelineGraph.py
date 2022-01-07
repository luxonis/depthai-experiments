import sys
import re
import json

import depthai as dai

class DAIPipelineGraph:

    def __init__(self, path):

        # Setup pipeline
        self.pipeline = dai.Pipeline()

        # Setup Node-Building Functions
        self.createNode = {
            "ColorCamera": self.CreateColorCamera,
            "EdgeDetector": self.CreateEdgeDetector,
            "FeatureTracker": self.CreateFeatureTracker,
            "ImageManip": self.CreateImageManip,
            "IMU": self.CreateIMU,
            "MobileNetDetectionNetwork": self.CreateMobileNetDetectionNetwork,
            "MobileNetSpatialDetectionNetwork": self.CreateMobileNetSpatialDetectionNetwork,
            "MonoCamera": self.CreateMonoCamera,
            "NeuralNetwork": self.CreateNeuralNetwork,
            "ObjectTracker": self.CreateObjectTracker,
            "Script": self.CreateScript,
            "SPIIn": self.CreateSPIIn,
            "SPIOut": self.CreateSPIOut,
            "StereoDepth": self.CreateStereoDepth,
            "SystemLogger": self.CreateSystemLogger,
            "VideoEncoder": self.CreateVideoEncoder,
            "XLinkIn": self.CreateXLinkIn,
            "XLinkOut": self.CreateXLinkOut,
            "YoloDetectionNetwork": self.CreateYoloDetectionNetwork,
            "YoloSpatialDetectionNetwork": self.CreateYoloSpatialDetectionNetwork,
        }

        # Parse JSON
        print( "DAIPipelineGraph: Creating pipeline from " + path )
        f = open( path )
        data = json.load( f )

        # Publicly accessible maps
        self.nodes = {}
        self.xout_streams = []

        # Temporary map that uses UIDs for connections
        node_map = {}

        # Create Nodes
        for node_id in data[ "nodes" ]:

            if data[ "nodes" ][ node_id ][ "disabled" ]:
                continue;

            node_data = data[ "nodes" ][ node_id ][ "custom" ]
            node_name = node_data[ "node_name" ]

            # Parse the type (and assume it starts with "depthai." and ends with "Node" )
            node_type = data[ "nodes" ][ node_id ][ "type_" ][8:-4]
            
            # Build the node
            if node_type in self.createNode:
                node_map[ node_id ] = self.createNode[ node_type ]( node_data )
                self.nodes[ node_name ] = node_map[ node_id ]
            else:
                print( "DAIPipelineGraph Error: Unknown Node Type: " + node_type + ", skipping it." )

        # Create connections
        if not "connections" in data:
            print( "DAIPipelineGraph: No connections found in Pipeline Graph" )
            return

        for connection in data[ "connections" ]:

            # Get Nodes. If this fails, the node wasn't registered and the connection is skipped.
            try:
                node_out = node_map[ connection[ "out" ][ 0 ] ]
                node_in = node_map[ connection[ "in" ][ 0 ] ]
            except:
                continue

            # Get output port. Special case for Script nodes.
            if isinstance( node_out, dai.node.Script ):
                port_out = node_out.outputs[ connection[ "out" ][ 1 ] ]
            else:
                port_out = getattr( node_out, connection[ "out" ][ 1 ] )

            # Get output port. Special case for Script nodes.
            if isinstance( node_in, dai.node.Script ):
                port_in = node_in.inputs[ connection[ "in" ][ 1 ] ]
            else:
                port_in = getattr( node_in, connection[ "in" ][ 1 ] )

            # Make connection
            port_out.link( port_in )


    # NODES
    def CreateColorCamera( self, node_data ):
        node = self.pipeline.create( dai.node.ColorCamera )
        node.setBoardSocket( getattr( dai.CameraBoardSocket, node_data[ "board_socket" ] ) )
        node.setImageOrientation( getattr( dai.CameraImageOrientation, node_data[ "orientation" ] ) )
        node.setResolution( getattr( dai.ColorCameraProperties.SensorResolution, node_data[ "resolution" ] ) )
        node.setInterleaved( node_data[ "interleaved" ] )
        return node

    def CreateMonoCamera( self, node_data ):
        node = self.pipeline.create( dai.node.MonoCamera )
        node.setBoardSocket( getattr( dai.CameraBoardSocket, node_data[ "board_socket" ] ) )
        node.setImageOrientation( getattr( dai.CameraImageOrientation, node_data[ "orientation" ] ) )
        node.setResolution( getattr( dai.MonoCameraProperties.SensorResolution, node_data[ "resolution" ] ) )
        return node

    def CreateStereoDepth( self, node_data ):
        node = self.pipeline.create( dai.node.StereoDepth )
        node.setLeftRightCheck( node_data[ "lr_check" ] )
        node.setExtendedDisparity( node_data[ "extended_disparity" ] )
        node.setSubpixel( node_data[ "subpixel" ] )
        node.initialConfig.setMedianFilter( getattr( dai.MedianFilter, node_data[ "median_filter" ] ) )
        node.initialConfig.setConfidenceThreshold( int( node_data[ "confidence_threshold" ] ) )
        return node

    def CreateEdgeDetector( self, node_data ):
        node = self.pipeline.create( dai.node.EdgeDetector )
        node.setWaitForConfigInput( node_data[ "wait_for_config_input" ] )
        return node

    def CreateFeatureTracker( self, node_data ):
        node = self.pipeline.create( dai.node.FeatureTracker )
        node.setHardwareResources( int( node_data[ "num_shaves" ] ), int( node_data[ "num_memory_slices" ] ) )
        node.setWaitForConfigInput( node_data[ "wait_for_config_input" ] )
        return node

    def CreateImageManip( self, node_data ):
        node = self.pipeline.create( dai.node.ImageManip )
        return node

    def CreateIMU( self, node_data ):
        node = self.pipeline.create( dai.node.IMU )

        sensors = []

        for key in node_data:
            if key not in [ "node_name", "report_rate", "batch_report_threshold", "max_batch_reports" ]:
                if node_data[ key ]:
                    sensors.append( getattr( dai.IMUSensor, key ) )

        node.enableIMUSensor( sensors, int( node_data[ "report_rate" ] ) )
        node.setBatchReportThreshold( int( node_data[ "batch_report_threshold" ] ) )
        node.setMaxBatchReports( int( node_data[ "max_batch_reports" ] ) )
        return node

    def CreateMobileNetDetectionNetwork( self, node_data ):
        node = self.pipeline.create( dai.node.MobileNetDetectionNetwork )
        node.setConfidenceThreshold( float( node_data[ "confidence_threshold" ] ) )
        return node

    def CreateMobileNetSpatialDetectionNetwork( self, node_data ):
        node = self.pipeline.create( dai.node.MobileNetSpatialDetectionNetwork )
        node.setConfidenceThreshold( float( node_data[ "confidence_threshold" ] ) )
        node.setBoundingBoxScaleFactor( float( node_data[ "bounding_box_scale" ] ) )
        node.setDepthLowerThreshold( int( node_data[ "lower_depth_threshold" ] ) )
        node.setDepthUpperThreshold( int( node_data[ "upper_depth_threshold" ] ) )
        return node

    def CreateNeuralNetwork( self, node_data ):
        node = self.pipeline.create( dai.node.NeuralNetwork )
        return node

    def CreateObjectTracker( self, node_data ):
        node = self.pipeline.create( dai.node.ObjectTracker )
        node.setTrackerType( getattr( dai.TrackerType, node_data[ "type" ] ) )
        node.setTrackerIdAssigmentPolicy( getattr( dai.TrackerIdAssigmentPolicy, node_data[ "id_policy" ] ) )
        node.setTrackerThreshold( float( node_data[ "threshold" ] ) )
        return node

    def CreateScript( self, node_data ):
        node = self.pipeline.create( dai.node.Script )
        return node

    def CreateSPIIn( self, node_data ):
        node = self.pipeline.create( dai.node.SPIIn )
        node.setStreamName( node_data[ "node_name" ] )
        node.setBusId( int( node_data[ "bus_id" ] ) )
        return node

    def CreateSPIOut( self, node_data ):
        node = self.pipeline.create( dai.node.SPIOut )
        node.setStreamName( node_data[ "node_name" ] )
        node.setBusId( int( node_data[ "bus_id" ] ) )
        return node

    def CreateSystemLogger( self, node_data ):
        node = self.pipeline.create( dai.node.SystemLogger )
        node.setRate( float( node_data[ "rate" ] ) )
        return node

    def CreateVideoEncoder( self, node_data ):
        node = self.pipeline.create( dai.node.VideoEncoder )
        node.setQuality( int( node_data[ "quality" ] ) )
        return node

    def CreateXLinkIn( self, node_data ):
        node = self.pipeline.create( dai.node.XLinkOut )
        node.setStreamName( node_data[ "node_name" ] )
        return node

    def CreateXLinkOut( self, node_data ):
        node = self.pipeline.create( dai.node.XLinkOut )
        node.setStreamName( node_data[ "node_name" ] )
        node.setFpsLimit( node_data[ "fps_limit" ] )
        node.setMetadataOnly( node_data[ "get_metadata_only" ] )
        self.xout_streams.append( node_data[ "node_name" ] )
        return node

    def CreateYoloDetectionNetwork( self, node_data ):
        node = self.pipeline.create( dai.node.YoloDetectionNetwork )
        node.setConfidenceThreshold( float( node_data[ "confidence_threshold" ] ) )
        node.setIouThreshold( float( node_data[ "iou_threshold" ] ) )
        node.setCoordinateSize( int( node_data[ "coordinate_size" ] ) )
        return node

    def CreateYoloSpatialDetectionNetwork( self, node_data ):
        node = self.pipeline.create( dai.node.YoloSpatialDetectionNetwork )
        node.setConfidenceThreshold( float( node_data[ "confidence_threshold" ] ) )
        node.setIouThreshold( float( node_data[ "iou_threshold" ] ) )
        node.setCoordinateSize( int( node_data[ "coordinate_size" ] ) )
        node.setBoundingBoxScaleFactor( float( node_data[ "bounding_box_scale" ] ) )
        node.setDepthLowerThreshold( int( node_data[ "lower_depth_threshold" ] ) )
        node.setDepthUpperThreshold( int( node_data[ "upper_depth_threshold" ] ) )
        return node