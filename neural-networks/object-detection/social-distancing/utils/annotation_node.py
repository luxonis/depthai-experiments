import depthai as dai


class AnnotationNode(dai.node.ThreadedHostNode):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.input_annotations = self.createInput()

    def run(self):
        while self.isRunning():
            # frame: dai.ImgFrame = self.input_frame.get()
            print("AnnotationNode")
            try:
                annotations_frame: dai.ImgFrame = self.input_annotations.get()
            except Exception as e:
                raise e

            # print(type(frame))
            print(type(annotations_frame))

            # annotations_frame = annotations_frame.getCvFrame()

            # new_frame = dai.ImgFrame()
            # new_frame.setCvFrame(annotations_frame, dai.ImgFrame.Type.BGR888i)
            # new_frame.setTimestamp(frame.getTimestamp())
            # new_frame.setSequenceNum(frame.getSequenceNum())
            # self.out_annotations.send(new_frame)

            # annotations_frame.setTimestamp(frame.getTimestamp())
            # annotations_frame.setSequenceNum(frame.getSequenceNum())

            # self.out_annotations.send(annotations_frame)
