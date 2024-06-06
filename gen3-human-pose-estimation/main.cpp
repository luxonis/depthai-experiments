// Includes common necessary includes for development using depthai library
#include "depthai/depthai.hpp"

#include <opencv2/opencv.hpp>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"

#include <tuple>

#include <sys/time.h>
#include <sys/resource.h>
#include <iostream>

class writeFPS : public dai::NodeCRTP<dai::node::HostNode, writeFPS> {
   private:
    std::chrono::steady_clock::time_point startTime;
    float fps = 0;
    int frames = 0;

   public:
    Input& input = inputs["in"];

    std::shared_ptr<writeFPS> build(Output& out) {
        startTime = std::chrono::steady_clock::now();
        frames = 0;
        fps = 0;

        out.link(input);
        return std::static_pointer_cast<writeFPS>(this->shared_from_this());
    }

    std::shared_ptr<dai::Buffer> processGroup(std::shared_ptr<dai::MessageGroup> in) override {
        frames++;
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
        if(elapsed > 1000) {
            fps = frames * 1000.0f / elapsed;
            frames = 0;
            startTime = currentTime;
        }

        auto inValue = in->get<dai::ImgFrame>("in");
        auto inFrame = inValue->getCvFrame();
        cv::putText(inFrame, "FPS: " + std::to_string(fps), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        auto frame = std::make_shared<dai::ImgFrame>();
        frame->setCvFrame(inFrame, dai::ImgFrame::Type::BGR888p);
        return frame;
    }
};

class Resize : public dai::NodeCRTP<dai::node::HostNode, Resize> {
   public:
    Input& input = inputs["in"];
    int windowWidth = 0;
    int windowHeight = 0;

    std::shared_ptr<Resize> build(Output& out, int width = 640, int height = 480) {
        windowWidth = width;
        windowHeight = height;
        out.link(input);
        return std::static_pointer_cast<Resize>(this->shared_from_this());
    }

    std::shared_ptr<dai::Buffer> processGroup(std::shared_ptr<dai::MessageGroup> in) override {
        auto inValue = in->get<dai::ImgFrame>("in");
        auto inFrame = inValue->getCvFrame();
        cv::resize(inFrame, inFrame, cv::Size(windowWidth, windowHeight));

        cv::Mat displayFrame(windowHeight, windowWidth, CV_8UC3);
        inFrame.copyTo(displayFrame);

        auto frame = std::make_shared<dai::ImgFrame>();
        frame->setCvFrame(displayFrame, dai::ImgFrame::Type::BGR888p);
        return frame;
    }
};

class Display : public dai::NodeCRTP<dai::node::HostNode, Display> {
   public:
    Input& input = inputs["in"];

    std::shared_ptr<Display> build(Output& out) {
        out.link(input);
        return std::static_pointer_cast<Display>(this->shared_from_this());
    }
    std::shared_ptr<dai::Buffer> processGroup(std::shared_ptr<dai::MessageGroup> in) override {
        auto frame = in->get<dai::ImgFrame>("in");
        cv::Mat frameCv = frame->getCvFrame();
        cv::imshow("Display", frameCv);
        auto key = cv::waitKey(1);
        if(key == 'q') {
            stopPipeline();
        }
        return nullptr;
    }
};

class  HumanPoseEstimationVisualizer : public dai::NodeCRTP<dai::node::HostNode, HumanPoseEstimationVisualizer> {
   private:
		const std::vector<std::vector<int> > COLORS = {
			{0, 100, 255}, 
			{0, 100, 255}, 
			{0, 255, 255}, 
			{0, 100, 255}, 
			{0, 255, 255}, 
			{0, 100, 255}, 
			{0, 255, 0},
            {255, 200, 100}, 
			{255, 0, 255}, 
			{0, 255, 0}, 
			{255, 200, 100}, 
			{255, 0, 255}, 
			{0, 0, 255}, 
			{255, 0, 0},
          	{200, 200, 0}, 
			{255, 0, 0}, 
			{200, 200, 0}, 
			{0, 0, 0}
		};

		const std::vector<std::pair<int, int> > POSE_PAIRS = {
            {1, 2}, {1, 5}, {2, 3}, {3, 4}, 
            {5, 6}, {6, 7}, {1, 8}, {8, 9}, 
            {9, 10}, {1, 11}, {11, 12}, {12, 13},
            {1, 0}, {0, 14}, {14, 16}, {0, 15}, 
            {15, 17}, {2, 17}, {5, 16}
        };

		const std::vector<std::pair<int, int> > MAPIDX = {
			{31, 32}, {39, 40}, {33, 34}, {35, 36}, 
			{41, 42}, {43, 44}, {19, 20}, {21, 22}, 
			{23, 24}, {25, 26}, {27, 28}, {29, 30}, 
			{47, 48}, {49, 50}, {53, 54}, {51, 52}, 
			{55, 56}, {37, 38}, {45, 46}
		};

		const std::vector<std::string> KEYPOINTMAPPINGS = {
			"Nose", "Neck", "R-Sho", "R-Elb", 
			"R-Wr", "L-Sho", "L-Elb", "L-Wr", 
			"R-Hip", "R-Knee", "R-Ank", "L-Hip", 
			"L-Knee", "L-Ank", "R-Eye", "L-Eye", 
			"R-Ear", "L-Ear"
		};

        // ! This ought to be slower than a better alternative but I cannot for the 
        // ! life of me figure out how to do it
        void turn2DxArrayToMat(const xt::xarray<float>& xarray, cv::Mat& mat){
            for (int i = 0; i < xarray.shape()[0]; i++){
                for (int j = 0; j < xarray.shape()[1]; j++){
                    mat.at<float>(i, j) = xarray(i, j);
                }
            }
        }


		std::vector<std::vector<std::tuple<cv::Point, float, int> > > DETECTED_KEYPOINTS;
        std::vector<std::tuple<cv::Point, float> > KEYPOINTS_LIST;
        std::vector<std::vector<int> > PERSONWISE_KEYPOINTS;



        int numberOfInterpolationSamples = 10;
        float paf_score_th = 0.2;
        float conf_th = 0.4;

        std::vector<std::pair<cv::Point, float> > getKeypoints(cv::Mat& probMap, double threshold = 0.2) {
            std::vector<std::pair<cv::Point, float> > keypoints;
            
            // cout current size
            cv::Size newSize = cv::Size(456, 256);

            // Resize the pronMap
            //! Not sure what resize alg is best here
            cv::resize(probMap, probMap, newSize, 0, 0, cv::INTER_NEAREST);

            // Smooth the probMap
            cv::GaussianBlur(probMap, probMap, cv::Size(3, 3), 0, 0);

            // Threshold the probMap
            cv::Mat probMapThreshold = cv::Mat::zeros(probMap.size(), CV_32F);
            cv::threshold(probMap, probMapThreshold, threshold, 1.0, cv::THRESH_BINARY);
            probMapThreshold.convertTo(probMapThreshold, CV_8UC1, 255);

            // Find contours
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;
            cv::findContours(probMapThreshold, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            // loop through contours
            for (int i = 0; i < contours.size(); i++) {
                cv::Mat blobMask = cv::Mat::zeros(probMap.size(), CV_32F);
                cv::fillPoly(blobMask, contours[i], cv::Scalar(1.0));
                cv::Mat maskedProbMap = probMap.mul(blobMask);

                double minVal, maxVal;
                cv::Point minLoc, maxLoc;
                cv::minMaxLoc(maskedProbMap, &minVal, &maxVal, &minLoc, &maxLoc);

                keypoints.push_back(std::make_pair(maxLoc, maxVal));
            }


            return keypoints;
        }
                    
        std::pair<
            std::vector<std::vector<std::tuple<int, int, float> > >,
            std::vector<int> 
        > getValidPairs(
            const xt::xarray<float>& outputs, int w, int h, 
            std::vector<std::vector<std::tuple<cv::Point, float, int> > >& detected_keypoints){
                // INPUT
                //    xArray<float> output (1 x 57 x 32 x 57)
                //    int w (456)
                //    int h (256)
                //    vector<vector<tuple<Point, float, int> > >& detected_keypoints (18 x [Point x Confidence x KeypointID])
                //
                // OUTPUT (pair)
                //    vector<vector<tuple<int, int, float> > > validPairs
                //    vector<int> invalidPairs

                auto validPairs = std::vector<std::vector<std::tuple<int, int, float> > >();
                auto invalidPairs = std::vector<int>();


                auto new_size = cv::Size(456, 256);

                for (int keypointPairIndedx = 0; keypointPairIndedx < MAPIDX.size(); keypointPairIndedx++){
                    auto pafX = xt::view(outputs, 0, MAPIDX[keypointPairIndedx].first, xt::all(), xt::all());
                    auto pafY = xt::view(outputs, 0, MAPIDX[keypointPairIndedx].second, xt::all(), xt::all());

                    cv::Mat pafXMat = cv::Mat::zeros(32, 57, CV_32F);
                    cv::Mat pafYMat = cv::Mat::zeros(32, 57, CV_32F);

                    turn2DxArrayToMat(pafX, pafXMat);
                    turn2DxArrayToMat(pafY, pafYMat);

                    // resize them 
                    // ! Same problem of unsure about the alg
                    cv::resize(pafXMat, pafXMat, new_size, 0, 0, cv::INTER_NEAREST);
                    cv::resize(pafYMat, pafYMat, new_size, 0, 0, cv::INTER_NEAREST);

                    auto candidatesA = detected_keypoints[POSE_PAIRS[keypointPairIndedx].first];
                    auto candidatesB = detected_keypoints[POSE_PAIRS[keypointPairIndedx].second];

                    int lenA = candidatesA.size();
                    int lenB = candidatesB.size();

                    // The candidate keypoints are not empty for either
                    if(lenA != 0 && lenB != 0){
                        auto valid_pair = std::vector<std::tuple<int, int, float> >();

                        for(int indexA = 0; indexA < lenA; indexA++){
                            int max_indexB = -1;
                            float maxScore = -1;
                            int found = 0;
                        
                            for(int indexB = 0; indexB < lenB; indexB++){
                                auto candidateA = candidatesA[indexA];
                                auto candidateB = candidatesB[indexB];

                                auto pointA = std::get<0>(candidateA);
                                auto pointB = std::get<0>(candidateB);

                                // We are looking at the direction of the vector (PAFs alg)
                                float vectorABx = pointB.x - pointA.x;
                                float vectorABy = pointB.y - pointA.y;
                                float normAB = std::sqrt(vectorABx * vectorABx + vectorABy * vectorABy);

                                if(normAB == 0){
                                    continue;
                                }
                                
                                vectorABx = vectorABx / normAB;
                                vectorABy = vectorABy / normAB;                            

                                int numberOfSamplesMoreThanThreshold = 0;
                                float sumOfPafScores = 0;
                                for (int i = 0; i < numberOfInterpolationSamples; i++){
                                    float alpha = (float)i / (numberOfInterpolationSamples - 1);
                                    auto point = pointA + alpha * (pointB - pointA);

                                    int x = (int)point.x;
                                    int y = (int)point.y;
                                    float pafx = pafXMat.at<float>(y, x);
                                    float pafy = pafYMat.at<float>(y, x);
                                    
                                    // ! Added normalization, not sure if this is ok
                                    float normPaf = std::sqrt(pafx * pafx + pafy * pafy);
                                    if(normPaf < 1e-8){
                                        continue;
                                    }
                                    pafx = pafx / normPaf;
                                    pafy = pafy / normPaf;

                                    // Calc DOT
                                    float pafScore = pafx * vectorABx + pafy * vectorABy;

                                    if(pafScore > paf_score_th){
                                        numberOfSamplesMoreThanThreshold++;
                                    }
                                    sumOfPafScores += pafScore;
                                }
                                float averagePafScore = sumOfPafScores / numberOfInterpolationSamples;

                                if(numberOfSamplesMoreThanThreshold / numberOfInterpolationSamples > conf_th){
                                    if(averagePafScore > maxScore){
                                        max_indexB = indexB;
                                        maxScore = averagePafScore;
                                        found = 1;
                                    }
                                }
                            }
                        
                            if(found){
                                int firstValue = std::get<2>(candidatesA[indexA]);
                                int secondValue = std::get<2>(candidatesB[max_indexB]);
                                float score = maxScore;

                                auto new_row = std::make_tuple(firstValue, secondValue, score);
                                valid_pair.push_back(new_row);
                            }

                        }
                        validPairs.push_back(valid_pair);
                    }else{
                        validPairs.push_back(std::vector<std::tuple<int, int, float> >());
                        invalidPairs.push_back(keypointPairIndedx);
                    }
                }  


                return std::make_pair(validPairs, invalidPairs);

        }


        std::vector<std::vector<int> > getPersonwiseKeypoints(
            std::vector<std::vector<std::tuple<int, int, float> > > validPairs,
            std::vector<int> invalidPairs,
            std::vector<std::tuple<cv::Point, float> > keypoints_list ){

            // INPUT
            //    vector<vector<tuple<int, int, float> > > validPairs
            //    vector<int> invalidPairs
            //    vector<tuple<Point, float> > keypoints_list
            
            std::vector<std::vector<int> > personwiseKeypoints = std::vector<std::vector<int> >();

            for (int keypointPairIndedx = 0; keypointPairIndedx < MAPIDX.size(); keypointPairIndedx++){
                // check if the keypointPairIndex is in the invalidPairs
                if(std::find(invalidPairs.begin(), invalidPairs.end(), keypointPairIndedx) != invalidPairs.end()){
                    continue;
                }

                std::vector<int> keyPointsA = std::vector<int>();
                std::vector<int> keyPointsB = std::vector<int>();
                std::vector<float> scores = std::vector<float>();

                for (int i = 0; i < validPairs[keypointPairIndedx].size(); i++){
                    auto validPair = validPairs[keypointPairIndedx][i];
                    keyPointsA.push_back(std::get<0>(validPair));
                    keyPointsB.push_back(std::get<1>(validPair));
                    scores.push_back(std::get<2>(validPair));
                }
                int indexA = POSE_PAIRS[keypointPairIndedx].first;
                int indexB = POSE_PAIRS[keypointPairIndedx].second;

                for(int pairIndex = 0; pairIndex < validPairs[keypointPairIndedx].size(); pairIndex++){
                    int found = 0;
                    int person_idx = -1;
                    
                    for(int personwiseKeypointsIndex = 0; personwiseKeypointsIndex < personwiseKeypoints.size(); personwiseKeypointsIndex++){
                        auto personwiseKeypoint = personwiseKeypoints[personwiseKeypointsIndex];
                        if(personwiseKeypoint[indexA] == keyPointsA[pairIndex]){
                            person_idx = personwiseKeypointsIndex;
                            found = 1;
                            break;
                        }
                    }


                    if(found){
                        personwiseKeypoints[person_idx][indexB] = keyPointsB[pairIndex];
                        personwiseKeypoints[person_idx][18] += std::get<1>(keypoints_list[indexB]) + scores[pairIndex];
                    }else if(!found && keypointPairIndedx < 17){
                        auto row = std::vector<int>();
                        for(int i = 0; i < 19; i++) row.push_back(-1);
                        row[indexA] = keyPointsA[pairIndex];
                        row[indexB] = keyPointsB[pairIndex];
                        row[18] = std::get<1>(keypoints_list[indexA]) + std::get<1>(keypoints_list[indexB]) + scores[pairIndex];
                        personwiseKeypoints.push_back(row);
                    }
                }
            }

            return personwiseKeypoints;

        }

        void processTensors(xt::xarray<float>& heatmaps, xt::xarray<float>& pafs) {
            auto outputs_tensor = xt::concatenate(xt::xtuple(heatmaps, pafs), 1); // 1 x 57 x 32 x 57
            
            std::vector<std::tuple<cv::Point, float> > new_keypoints_list; // [Point x Confidence]
            std::vector<std::vector<std::tuple<cv::Point, float, int> > > new_keypoints; // [ [Point x Confidence x KeypointID] ]
            int keypoint_id = 0;

            // Each of the 18 points the BLOB is looking for
            for(int row = 0; row < 18; row++){
                cv::Mat probMaps = cv::Mat::zeros(32, 57, CV_32F);
                turn2DxArrayToMat(xt::view(heatmaps, 0, row, xt::all(), xt::all()), probMaps);

                auto keypoints = getKeypoints(probMaps, 0.3);
                std::vector<std::tuple<cv::Point, float, int> > keypoints_with_id;

                for (int i = 0; i < keypoints.size(); i++){
                    auto keypoint_point = std::get<0>(keypoints[i]);
                    auto keypoint_confidence = std::get<1>(keypoints[i]);

                    new_keypoints_list.push_back(std::make_tuple(keypoint_point, keypoint_confidence));
                    keypoints_with_id.push_back(std::make_tuple(keypoint_point, keypoint_confidence, keypoint_id));

                    keypoint_id++;
                }

                new_keypoints.push_back(keypoints_with_id);
            }

            // Get the valid pairs
            auto getValidPairsReturnValue = getValidPairs(outputs_tensor, 456, 256, new_keypoints);
            auto validPairs = getValidPairsReturnValue.first;
            auto invalidPairs = getValidPairsReturnValue.second;

            // Get the personwise keypoints
            auto newPersonwiseKeypoints = getPersonwiseKeypoints(validPairs, invalidPairs, new_keypoints_list);


            DETECTED_KEYPOINTS = new_keypoints;
            KEYPOINTS_LIST = new_keypoints_list;
            PERSONWISE_KEYPOINTS = newPersonwiseKeypoints;
        }

    void drawOnFrame(cv::Mat& frame) {
        for(int i = 0; i < 18; i++){
            for(int j = 0; j < DETECTED_KEYPOINTS[i].size(); j++){
                auto point = std::get<0>(DETECTED_KEYPOINTS[i][j]);
                auto color = cv::Scalar(COLORS[i][0], COLORS[i][1], COLORS[i][2]);
                cv::circle(frame, point, 5, color, -1);
            }
        }
        for(int i = 0; i < 17; i++){
            for(int j = 0; j < PERSONWISE_KEYPOINTS.size(); j++){
                int indexA = PERSONWISE_KEYPOINTS[j][POSE_PAIRS[i].first];
                int indexB = PERSONWISE_KEYPOINTS[j][POSE_PAIRS[i].second];

                if(indexA != -1 && indexB != -1){
                    auto pointA = std::get<0>(KEYPOINTS_LIST[indexA]);
                    auto pointB = std::get<0>(KEYPOINTS_LIST[indexB]);
                    auto color = cv::Scalar(COLORS[i][0], COLORS[i][1], COLORS[i][2]);

                    cv::line(frame, pointA, pointB, color, 2);
                }
            }
        }
    }

   public:
	Input& inPassthrough = inputs["passthrough"];
	Input& inDataNN = inputs["NNData"];

	std::shared_ptr<HumanPoseEstimationVisualizer> build(Output& passThrough, Output& input) {
        // print out the lengths of COLORS, POSE_PAIRS, MAPIDX, and KEYPOINTMAPPINGS
        std::cout << "COLORS: " << COLORS.size() << std::endl;
        std::cout << "POSE_PAIRS: " << POSE_PAIRS.size() << std::endl;
        std::cout << "MAPIDX: " << MAPIDX.size() << std::endl;
        std::cout << "KEYPOINTMAPPINGS: " << KEYPOINTMAPPINGS.size() << std::endl;


		passThrough.link(inPassthrough);
		input.link(inDataNN);

		return std::static_pointer_cast<HumanPoseEstimationVisualizer>(this->shared_from_this());
	}

	std::shared_ptr<dai::Buffer> processGroup(std::shared_ptr<dai::MessageGroup> in) override {
		auto frame = in->get<dai::ImgFrame>("passthrough")->getCvFrame();

		auto nnData = in->get<dai::NNData>("NNData");		
		
		const std::string pafLayerName = "Mconv7_stage2_L1";
		const std::string heatmapsLayerName = "Mconv7_stage2_L2";

		auto pafLayer = nnData->getTensor<float>(pafLayerName); // 1 x 38 x 32 x 57
		auto heatmapsLayer = nnData->getTensor<float>(heatmapsLayerName); // 1 x 19 x 32 x 57

        processTensors(heatmapsLayer, pafLayer);
        drawOnFrame(frame);
		
		auto returnFrame = std::make_shared<dai::ImgFrame>();
        returnFrame->setCvFrame(frame, dai::ImgFrame::Type::BGR888p);
		return returnFrame;
	}
};

int main() {
    std::cout << "BLOB_PATH: " << BLOB_PATH << std::endl;

      dai::Path path = dai::Path(BLOB_PATH);
    dai::OpenVINO::Blob blob = dai::OpenVINO::Blob(path);

    // Create pipeline
    dai::Pipeline pipeline(true);

    // Define source and output
    auto camRgb = pipeline.create<dai::node::ColorCamera>();

    // Properties
    camRgb->setBoardSocket(dai::CameraBoardSocket::CAM_A);
    camRgb->setResolution(dai::ColorCameraProperties::SensorResolution::THE_1080_P);
    camRgb->setFps(37);
    camRgb->setVideoSize(1920, 1080);
	
    auto resizeSmall = pipeline.create<Resize>()->build(camRgb->video, 456, 256);

	auto hpeNN = pipeline.create<dai::node::NeuralNetwork>();

	hpeNN->setBlob(blob);
	hpeNN->input.setBlocking(false);
	hpeNN->setNumInferenceThreads(2);
	hpeNN->setBlobPath(BLOB_PATH);

	resizeSmall->out.link(hpeNN->input);
	auto hpeOut = pipeline.create<HumanPoseEstimationVisualizer>()->build(hpeNN->passthrough, hpeNN->out);

    auto resizeBig = pipeline.create<Resize>()->build(hpeOut->out, 912, 512);

    auto writeFps = pipeline.create<writeFPS>()->build(resizeBig->out);
	auto display = pipeline.create<Display>()->build(writeFps->out);


    pipeline.start();
    pipeline.wait();
    return 0;


}

