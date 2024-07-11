// Sample
/*
// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"

int main() {
    // Create pipeline
    dai::Pipeline pipeline(true);

    // Define source and output
    auto camRgb = pipeline.create<dai::node::ColorCamera>();

    // Properties
    camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setVideoSize(1920, 1080);

    auto outputQueue = camRgb->video.createOutputQueue();

    pipeline.start();
    while(pipeline.isRunning()) {
        auto videoIn = outputQueue->get<dai::ImgFrame>();

        // Get BGR frame from NV12 encoded video frame to show with opencv
        // Visualizing the frame on slower hosts might have overhead
        cv::imshow("video", videoIn->getCvFrame());

        int key = cv::waitKey(1);
        if(key == 'q' || key == 'Q') {
            pipeline.stop();
        }
    }
    return 0;
}
*/

#include <fstream>
#include <iostream>
#include <vector>
#include "depthai/depthai.hpp"
//#include "blobconverter/"

int main(){
    dai::Pipeline pipeline(true);
    pipeline.setOpenVINOVersion(dai::OpenVINO::VERSION_2021_4);
    std::string openvino_version = "2021.4";
    std::tuple<int,int> VIDEO_SIZE = {1072,1072};

    auto cam = pipeline.create<dai::node::ColorCamera>();
    // For ImageManip rotate you need input frame of multiple of 16
    cam->setPreviewSize(1072,1072);
    cam->setVideoSize(VIDEO_SIZE);
    cam->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    cam->setInterleaved(false);
    cam->setPreviewNumFramesPool(20);
    cam->setFps(20);
    cam->setBoardSocket(dai::CameraBoardSocket::CAM_A);


    std::map<std::string,std::shared_ptr<dai::MessageQueue>> queues;
    queues["color"] = cam->video.createOutputQueue();

    // ImageManip that will crop the frame before sending it to the Face detection NN node
    auto face_det_manip = pipeline.create<dai::node::ImageManip>();
    face_det_manip->initialConfig.setResize(300,300);
    face_det_manip->setMaxOutputFrameSize(300*300*3);
    cam->preview.link(face_det_manip->inputImage);

    //=================[ FACE DETECTION ]=================
    std::cout<<"Creating Face Detection Neural Network..."<<std::endl;
    auto face_det_nn = pipeline.create<dai::node::MobileNetDetectionNetwork>();
    face_det_nn->setConfidenceThreshold(0.5);
    /*face_det_nn->setBlobPath(blobconverter.from_zoo(
    name="face-detection-retail-0004",
    shaves=6,
    version=openvino_version)
    );*/

    // Link Face ImageManip -> Face detection NN node
    face_det_manip->out.link(face_det_nn->input);

    queues["detection"] = face_det_nn->out.createOutputQueue();

    //=================[ SCRIPT NODE ]=================
    // Script node will take the output from the face detection NN as an input and set ImageManipConfig
    // to the 'age_gender_manip' to crop the initial frame
    auto script = pipeline.create<dai::node::Script>();
    script->setProcessor(dai::ProcessorType::LEON_CSS);

    face_det_nn->out.link(script->inputs["face_det_in"]); 
    face_det_nn->passthrough.link(script->inputs["face_pass"]);

    cam->preview.link(script->inputs["preview"]);
    
    std::fstream f("script.py",std::ios::binary);
    //script->setScript(f.read());
    
    //=================[ HEAD POSE ESTIMATION ]=================
    auto headpose_manip = pipeline.create<dai::node::ImageManip>();
    headpose_manip->initialConfig.setResize(60,60);
    script->outputs["headpose_cfg"].link(headpose_manip->inputConfig);
    script->outputs["headpose_img"].link(headpose_manip->inputImage);

    auto headpose_nn = pipeline.create<dai::node::NeuralNetwork>();
    //headpose_nn->setBlobPath(blobconverter.from_zoo(
    //    name="head-pose-estimation-adas-0001",
    //    shaves=6,
    //    version=openvino_version
    //));
    headpose_manip->out.link(headpose_nn->input);

    headpose_nn->out.link(script->inputs["headpose_in"]);    
    headpose_nn->passthrough.link(script->inputs["headpose_pass"]);

    //=================[ LANDMARKS DETECTION ]=================
    auto landmark_manip = pipeline.create<dai::node::ImageManip>();
    landmark_manip->initialConfig.setResize(48,48);
    script->outputs["landmark_cfg"].link(landmark_manip->inputConfig);
    script->outputs["landmark_img"].link(landmark_manip->inputImage);

    auto landmark_nn = pipeline.create<dai::node::NeuralNetwork>();
    //landmark_nn->setBlobPath(blobconverter.from_zoo(
    //name="landmarks-regression-retail-0009",
    //shaves=6,
    //version=openvino_version
    //));
    
    landmark_manip->out.link(landmark_nn->input);

    landmark_nn->out.link(script->inputs["landmark_in"]);
    landmark_nn->passthrough.link(script->inputs["landmark_pass"]);

    queues["landmarks"] = landmark_nn->out.createOutputQueue();

    //=================[ LEFT EYE CROP ]=================
    auto left_manip = pipeline.create<dai::node::ImageManip>();
    left_manip->initialConfig.setResize(60,60);
    left_manip->inputConfig.setWaitForMessage(true);
    script->outputs["left_manip_img"].link(left_manip->inputImage);
    script->outputs["left_manip_img"].link(left_manip->inputConfig);
    left_manip->out.link(script->inputs["left_eye_in"]);

    //=================[ RIGHT EYE CROP ]=================
    auto right_manip = pipeline.create<dai::node::ImageManip>();
    right_manip->initialConfig.setResize(60,60);
    right_manip->inputConfig.setWaitForMessage(true);
    script->outputs["right_manip_img"].link(right_manip->inputImage);
    script->outputs["right_manip_img"].link(right_manip->inputConfig);
    right_manip->out.link(script->inputs["right_eye_in"]);

    //=================[ GAZE ESTIMATION ]=================

    auto gaze_nn = pipeline.create<dai::node::NeuralNetwork>();
    //gaze_nn.setBlobPath(blobconverter.from_zoo(
    //    name="gaze-estimation-adas-0002",
    //    shaves=6,
    //    version=openvino_version,
    //    compile_params=['-iop head_pose_angles:FP16,right_eye_image:U8,left_eye_image:U8']
    //))

    std::vector<std::string> SCRIPT_OUTPUT_NAMES = {"to_gaze_head","to_gaze_left","to_gaze_right"},
    NN_NAMES = {"head_pose_angles","left_eye_image","right_eye_image"};

    for(int i=0;i<SCRIPT_OUTPUT_NAMES.size();i++){
        auto script_name = SCRIPT_OUTPUT_NAMES[i];
        auto nn_name = NN_NAMES[i];
        // Link Script node output to NN input
        script->outputs[script_name].link(gaze_nn->inputs[nn_name]);
        // Set NN input to blocking and to not reuse previous msgs
        gaze_nn->inputs[nn_name].setBlocking(true);
        gaze_nn->inputs[nn_name].setReusePreviousMessage(false);

    }
    //# Workaround, so NNData (output of gaze_nn) will take seq_num from this message (FW bug)
    //# Will be fixed in depthai 2.24
    //gaze_nn.passthroughs['left_eye_image'].link(script.inputs['none'])
    //script.inputs['none'].setBlocking(False)
    //script.inputs['none'].setQueueSize(1)
    //
    queues["gaze"] = gaze_nn->out.createOutputQueue();

    //==================================================

    pipeline.start();
    while(pipeline.isRunning()) {

        while(true){
            std::vector<std::string> names = {"color","detection","landmarks","gaze"};
            for(auto name : names){
                if(queues[name]->has()){
                    auto msg = queues[name]->get();
                    if(name == "color"){
                        auto videoIn = queues["color"]->get<dai::ImgFrame>();
                        cv::imshow("video",videoIn->getCvFrame());
                    }
                }

            }


        /*
        msgs = sync.get_msgs()
        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            dets = msgs["detection"].detections
            for i, detection in enumerate(dets):
                det = BoundingBox(detection)
                tl, br = det.denormalize(frame.shape)
                cv2.rectangle(frame, tl, br, (10, 245, 10), 1)

                gaze = np.array(msgs["gaze"][i].getFirstLayerFp16())
                gaze_x, gaze_y = (gaze * 100).astype(int)[:2]

                landmarks = np.array(msgs["landmarks"][i].getFirstLayerFp16())
                colors = [(0, 127, 255), (0, 127, 255), (255, 0, 127), (127, 255, 0), (127, 255, 0)]
                for lm_i in range(0, len(landmarks) // 2):
                    # 0,1 - left eye, 2,3 - right eye, 4,5 - nose tip, 6,7 - left mouth, 8,9 - right mouth
                    x, y = landmarks[lm_i*2:lm_i*2+2]
                    point = det.map_point(x,y).denormalize(frame.shape)

                    if lm_i <= 1: # Draw arrows from left eye & right eye
                        cv2.arrowedLine(frame, point, ((point[0] + gaze_x*5), (point[1] - gaze_y*5)), colors[lm_i], 3)

                    cv2.circle(frame, point, 2, colors[lm_i], 2)

            cv2.imshow("Lasers", frame)*/

            int key = cv::waitKey(1);
            if(key == 'q' || key == 'Q') {
                pipeline.stop();
                break;
            }
        }
    }
    return 0;

}