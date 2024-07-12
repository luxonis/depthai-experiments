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
#include "MultiMsgSync.cpp"
#include "bbox.cpp"

std::size_t getTensorDataSize(const TensorInfo& tensor) {
    uint32_t i;

    // Use the first non zero stride
    for(i = 0; i < tensor.strides.size(); i++) {
        if(tensor.strides[i] > 0) {
            break;
        }
    }
    return tensor.dims[i] * tensor.strides[i];
}

 std::vector<float> getData(dai::TensorInfo &tensor, std::shared_ptr<dai::NNData> dat) {
    if(tensor.dataType == dai::TensorInfo::DataType::FP16) {
        // Total data size = last dimension * last stride
        if(tensor.numDimensions > 0) {
            std::size_t size = getTensorDataSize(tensor);
            std::size_t numElements = size / 2;  // FP16

            std::vector<float> data;
            data.reserve(numElements);
            auto* pFp16Data = reinterpret_cast<std::uint16_t*>(dat->data->getData()[tensor.offset]);
            for(std::size_t i = 0; i < numElements; i++) {
                data.push_back(fp16_ieee_to_fp32_value(pFp16Data[i]));
            }
            return data;
        }
    } else if(tensor.dataType == dai::TensorInfo::DataType::FP32) {
        if(tensor.numDimensions > 0) {
            std::size_t size = getTensorDataSize(tensor);
            std::size_t numElements = size / sizeof(float_t);

            std::vector<float> data;
            data.reserve(numElements);
            auto* pFp32Data = reinterpret_cast<float_t*>(dat->data->getData()[tensor.offset]);
            for(std::size_t i = 0; i < numElements; i++) {
                data.push_back(pFp32Data[i]);
            }
            return data;
            }
        }
    return {};
 }



int main(){
    dai::Pipeline pipeline(true);
    pipeline.setOpenVINOVersion(dai::OpenVINO::VERSION_2021_4);
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
    face_det_nn->setBlobPath("face-detection-retail-0004.blob");

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
    
    std::ifstream f("script.py", std::ios::binary);
    std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(f), {});
    script->setScript(buffer);
    
    //=================[ HEAD POSE ESTIMATION ]=================
    auto headpose_manip = pipeline.create<dai::node::ImageManip>();
    headpose_manip->initialConfig.setResize(60,60);
    script->outputs["headpose_cfg"].link(headpose_manip->inputConfig);
    script->outputs["headpose_img"].link(headpose_manip->inputImage);

    auto headpose_nn = pipeline.create<dai::node::NeuralNetwork>();
    headpose_nn->setBlobPath("head-pose-estimation-adas-0001.blob");
    headpose_manip->out.link(headpose_nn->input);

    headpose_nn->out.link(script->inputs["headpose_in"]);    
    headpose_nn->passthrough.link(script->inputs["headpose_pass"]);

    //=================[ LANDMARKS DETECTION ]=================
    auto landmark_manip = pipeline.create<dai::node::ImageManip>();
    landmark_manip->initialConfig.setResize(48,48);
    script->outputs["landmark_cfg"].link(landmark_manip->inputConfig);
    script->outputs["landmark_img"].link(landmark_manip->inputImage);

    auto landmark_nn = pipeline.create<dai::node::NeuralNetwork>();
    landmark_nn->setBlobPath("landmarks-regression-retail-0009.blob");
    
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
    gaze_nn->setBlobPath("gaze-estimation-adas-0002.blob");

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
        TwoStageHostSeqSync sync;
        while(true){
            int key = cv::waitKey(1);
            if(key == 'q' || key == 'Q') {
                pipeline.stop();
                break;
            }

            std::vector<std::string> names = {"color","detection","landmarks","gaze"};
            for(auto name : names){
                if(queues[name]->has()){
                    auto msg = std::shared_ptr<dai::MessageQueue>(queues[name]);
                    sync.add_msg(msg,name);
                    if(name == "color"){
                        auto videoIn = queues["color"]->get<dai::ImgFrame>();
                        cv::imshow("video",videoIn->getCvFrame());
                    }
                }

            }

        auto msgs = sync.get_msgs();
        if(msgs.second == -1) continue;

        auto frame = msgs.first["color"][0]->get<dai::ImgFrame>()->getCvFrame();
        auto dets = msgs.first["detection"][0]->get<dai::ImgDetections>()->detections;
        
        for(int i = 0; i < dets.size();i++){
            auto &detection = dets[i];
            BoundingBox det(detection);
            // shape = height,width,channels
            // seems to be (1072,1072,3)
            //auto [tl,br] = det.denormalize(frame.shape);
            //replaced top-left and bottom-right with one array (easier impl)
            auto pts = det.denormalize({1072,1072,3});

            cv::rectangle(frame, cv::Point(pts[0],pts[1]),cv::Point(pts[2],pts[3]),
            cv::Scalar(10,245,10),1);           
            //cv2.rectangle(frame, tl, br, (10, 245, 10), 1)
            std::vector<float> gaze,landmarks;

            auto gaze_ptr = msgs.first["gaze"][i]->get<dai::NNData>();
            dai::TensorInfo gaze_info;
            gaze_ptr->getLayer(gaze_ptr->getAllLayerNames()[0],gaze_info);
            gaze = getData(gaze_info,gaze_ptr);

            auto gaze_x = (int)gaze[0], gaze_y = (int)gaze[1];

            auto landmarks_ptr = msgs.first["landmarks"][i]->get<dai::NNData>();
            dai::TensorInfo landmarks_info;
            auto landmarks = landmarks_ptr->getTensor<float>(landmarks_ptr->getAllLayerNames()[0],0);
            

            int colors[5][3] = { {0,127,255}, {0,127,255}, {255,0,127}, {127,255,0}, {127,255,0} };            
            for(int lm_i = 0;i < landmarks.size()/2;lm_i++){
                // 0,1 - left eye, 2,3 - right eye, 4,5 - nose tip, 6,7 - left mouth, 8,9 - right mouth
                auto x = landmarks[lm_i*2], y = landmarks[lm_i*2+1];
                // again,should be frame.shape
                auto point = det.map_point(x,y).denormalize({1072,1072,3});
                if(lm_i <= 1){ // Draw arrows from left eye & right eye
                    cv::arrowedLine(frame, cv::Point(point[0],point[1]), cv::Point((point[0] + gaze_x*5), (point[1] - gaze_y*5)), cv::Scalar(colors[lm_i][0],colors[lm_i][1],colors[lm_i][2]), 3);
                }
                cv::circle(frame,cv::Point(point[0],point[1]),2,cv::Scalar(colors[lm_i][0],colors[lm_i][1],colors[lm_i][2]),2);
            }

        }
        
        cv::imshow("Lasers",frame);
        }
    }
    return 0;

}