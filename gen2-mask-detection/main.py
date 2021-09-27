from scipy.special import log_softmax,softmax
from queue import Queue
from depthai_utils import *

class Main(DepthAI):
    def __init__(self,file=None,camera=False):
        self.face_coords = Queue()
        self.face_frame = Queue()
        self.cam_size = (300,300)
        super(Main,self).__init__(file,camera)

    def create_nns(self):
        self.create_mobilenet_nn("models/face-detection-retail-0004_openvino_2020_1_4shave.blob","face",True)
        self.create_models("models/sbd_mask_openvino_2020_1_4shave.blob","mask")
    
    def start_nns(self):
        if not self.camera:
            self.face_in = self.device.getInputQueue("face_in",4,False)
        self.face_nn = self.device.getOutputQueue("face_nn",4,False)
        self.mask_in = self.device.getInputQueue("mask_in",4,False)
        self.mask_nn = self.device.getOutputQueue("mask_nn",4,False)
    
    
    def run_face(self):
        if not self.camera:
            nn_data = run_nn(self.face_in,self.face_nn,{"data":to_planar(self.frame,self.get_cam_size)})
        else:
            nn_data = self.face_nn.tryGet()
        if nn_data is None:
            return False
        bboxes = nn_data.detections
        for bbox in bboxes:
            coords = frame_norm(self.frame,*[bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax])
            self.face_coords.put(coords)
            self.face_frame.put(self.frame[coords[1]:coords[3],coords[0]:coords[2]])
        return True
    
    def run_mask(self):
        try:
            count = self.face_frame.qsize()
            cnt_mask = 0
            while self.face_frame.qsize():
                face_coord = self.face_coords.get()
                face_frame = self.face_frame.get()
                nn_data = run_nn(self.mask_in,self.mask_nn,{"data":to_planar(face_frame,(224,224))})
                if nn_data is None:
                    return
                self.fps.update()
                out = to_nn_result(nn_data)
                match = softmax(np.array(out))
                index = np.argmax(match)
                ftype = 0 if index > 0.5 else 1
                color = (0,0,255) if ftype else (0,255,0)
                self.draw_bbox(face_coord,color)
                cv2.putText(self.debug_frame,'{:.2f}'.format(match[0]),(face_coord[0],face_coord[1]-10),cv2.FONT_HERSHEY_COMPLEX,1,color)
                if ftype == 0: cnt_mask += 1
            proportion = cnt_mask / count * 100
            cv2.putText(self.debug_frame,"masks:" + str(round(proportion,2)) + "%",(10,30),cv2.FONT_HERSHEY_COMPLEX,0.75,(255,0,0))
        except:
            pass
    
    def parse_run(self):
        face_success = self.run_face()
        if face_success:
            self.run_mask()

if __name__ == '__main__':
    if args.video:
        Main(file=args.video).run()
    else:
        Main(camera=args.camera).run()