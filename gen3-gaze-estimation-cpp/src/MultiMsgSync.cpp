// Color frames (ImgFrame), object detection (ImgDetections) and age/gender gaze (NNData)
// messages arrive to the host all with some additional delay.
// For each ImgFrame there's one ImgDetections msg, which has multiple detections, and for each
// detection there's a NNData msg which contains age/gender gaze results.//
// How it works:
// Every ImgFrame, ImgDetections and NNData message has it's own sequence number, by which we can sync messages.

#include <map>
#include <string>
#include <utility>
#include <depthai/depthai.hpp>


class TwoStageHostSeqSync{
    public:
    // name: color,detection or gaze
    void add_msg(std::shared_ptr<dai::MessageQueue> msg, std::string name){
        auto seq = std::to_string(msg->get<dai::NNData>()->getSequenceNum()); // getSequenceNum() doesnt exist 
        // print(f"Got {name}, seq: {seq}")
        
        if(name == "gaze" || name == "landmarks"){
            // Append msg to an array
            msgs[seq][name].push_back(msg);
            // print(f'Added gaze seq {seq}, total len {len(self.msgs[seq]["gaze"])}')
        }else if(name == "detection"){
            // Save detection msg in map
            msgs[seq][name].push_back(msg);
            // never used??
            //msgs[seq]["len"].push_back(msg->get<dai::ImgDetections>()->detections.size()); // also doesnt exist
        }else if(name == "color") {
            // Save color frame in map
            msgs[seq][name].push_back(msg); 
        }        
    }   

    std::pair<std::map<std::string,std::vector<std::shared_ptr<dai::MessageQueue>>>,int> get_msgs(){
        std::vector<std::string> seq_remove;
        for(auto [seq,r_msgs] : msgs){
            seq_remove.push_back(seq); // Will get removed from dict if we find synced msgs pairs
            // Check if we have both detections and color frame with this sequence number
            if(r_msgs.count("color") > 0 && r_msgs.count("len") > 0){
                // Check if all detected objects (faces) have finished gaze (age/gender) inference
                if(0 < r_msgs["gaze"].size()){
                    // print(f"Synced msgs with sequence number {seq}", msgs)
                    // We have synced msgs, remove previous msgs (memory cleaning)
                    for(auto rm : seq_remove){
                        msgs[rm].clear();
                    }
                    return {r_msgs,0}; // Returned synced msgs
                }
            }
        }
        return {msgs["color"],-1}; // No synced msgs
    }

    private:
        std::map<std::string,std::map<std::string,std::vector<std::shared_ptr<dai::MessageQueue>>>> msgs;
};