import depthai as dai


def get_copy(img_frame: dai.ImgFrame) -> dai.ImgFrame:
    copy = dai.ImgFrame()
    copy.setSequenceNum(img_frame.getSequenceNum())
    copy.setTimestamp(img_frame.getTimestamp())
    copy.setTimestampDevice(img_frame.getTimestampDevice())
    copy.setCvFrame(img_frame.getCvFrame(), img_frame.getType())
    return copy