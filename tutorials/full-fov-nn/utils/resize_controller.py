import depthai as dai


class ResizeController(dai.node.HostNode):
    def __init__(self):
        super().__init__()

        self.out_cfg = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImageManipConfigV2, True)
            ]
        )

        self.out_annotations = self.createOutput(
            possibleDatatypes=[
                dai.Node.DatatypeHierarchy(dai.DatatypeEnum.ImgAnnotations, True)
            ]
        )

    def build(
        self,
        frames: dai.Node.Output,
        nn_size: tuple[int, int],
        output_type: dai.ImgFrame.Type,
    ):
        self.nn_size = nn_size
        self.output_type = output_type
        self.current_mode = dai.ImageManipConfigV2.ResizeMode.STRETCH
        self.link_args(frames)
        return self

    # TODO: This is a temporary solution, until the bug is fixed in DepthAI. Remove this, once it's possible to
    # send multiple ImageManipConfigV2 messages with setOutputSize in a row.
    def send_dummy_config(self):
        self.out_cfg.send(dai.ImageManipConfigV2())

    def handle_key_press(self, key: int):
        cfg = dai.ImageManipConfigV2()
        cfg.setFrameType(self.output_type)

        if key == ord("a"):
            self.send_dummy_config()
            self.current_mode = dai.ImageManipConfigV2.ResizeMode.LETTERBOX
            cfg.setOutputSize(
                *self.nn_size, dai.ImageManipConfigV2.ResizeMode.LETTERBOX
            )
            self.out_cfg.send(cfg)
        elif key == ord("s"):
            self.send_dummy_config()
            self.current_mode = dai.ImageManipConfigV2.ResizeMode.STRETCH
            cfg.setOutputSize(*self.nn_size, dai.ImageManipConfigV2.ResizeMode.STRETCH)
            self.out_cfg.send(cfg)
        elif key == ord("d"):
            self.send_dummy_config()
            self.current_mode = dai.ImageManipConfigV2.ResizeMode.CENTER_CROP
            cfg.setOutputSize(
                *self.nn_size, dai.ImageManipConfigV2.ResizeMode.CENTER_CROP
            )
            self.out_cfg.send(cfg)

    def create_text_annot(self, text: str, pos: tuple[float, float]):
        txt_annot = dai.TextAnnotation()
        txt_annot.fontSize = 10
        txt_annot.backgroundColor = dai.Color(0, 1, 0, 1)
        txt_annot.textColor = dai.Color(1, 1, 1, 1)
        txt_annot.position = dai.Point2f(*pos)
        txt_annot.text = text
        return txt_annot

    def process(self, frame: dai.ImgFrame):
        img_annots = dai.ImgAnnotations()
        img_annot = dai.ImgAnnotation()
        selected_mode = self.create_text_annot(
            f"Selected resize mode: {self.current_mode.name}", (0.05, 0.1)
        )
        instruct1 = self.create_text_annot(
            "a - LETTERBOX (not yet supported on RVC4)", (0.05, 0.14)
        )
        instruct2 = self.create_text_annot("s - STRETCH", (0.05, 0.18))
        instruct3 = self.create_text_annot("d - CENTER_CROP", (0.05, 0.22))
        img_annot.texts.append(selected_mode)
        img_annot.texts.append(instruct1)
        img_annot.texts.append(instruct2)
        img_annot.texts.append(instruct3)
        img_annots.annotations.append(img_annot)
        img_annots.setTimestamp(frame.getTimestamp())
        self.out_annotations.send(img_annots)
