import depthai as dai


class GroupNNOutputs(dai.node.ThreadedHostNode):
    def __init__(self):
        super().__init__()
        