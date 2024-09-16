import csv
import depthai as dai

from pathlib import Path
from class_saver import ClassSaver


device = dai.Device()
platform = device.getPlatform().name

#model_description = dai.NNModelDescription(modelSlug="mobilenet-ssd", platform="RVC2", modelVersionSlug="300x300") # faster for RVC2
model_description = dai.NNModelDescription(modelSlug="yolov10-nano", platform=platform, modelVersionSlug="coco-512x288") 
archive_path = dai.getModelFromZoo(model_description)
nn_archive = dai.NNArchive(archive_path)

# Start defining a pipeline
with dai.Pipeline(device) as pipeline:

    cam_rgb = pipeline.create(dai.node.Camera).build(boardSocket=dai.CameraBoardSocket.CAM_A)
    rgb_preview = cam_rgb.requestOutput(size=(512, 288), type=dai.ImgFrame.Type.BGR888p)
    
    detection_nn = pipeline.create(dai.node.DetectionNetwork).build(input=rgb_preview, nnArchive=nn_archive, confidenceThreshold=0.5)

    # MobilenetSSD label texts
    texts = detection_nn.getClasses()
    
    data_folder_path = Path(__file__).parent / Path('data/')

    for text in texts:
        (data_folder_path / Path(text)).mkdir(parents=True, exist_ok=True)
    (data_folder_path/ Path("raw")).mkdir(parents=True, exist_ok=True)


    with open(data_folder_path / Path("dataset.csv'"), "w") as dataset_file:
        dataset = csv.DictWriter(
            dataset_file,
            ["timestamp", "label", "left", "top", "right", "bottom", "raw_frame", "overlay_frame", "cropped_frame"]
        )
        dataset.writeheader()

        class_saver = pipeline.create(ClassSaver).build(
            rgb=rgb_preview,
            nn_out=detection_nn.out,
            dataset=dataset,
            texts=texts)
        class_saver.set_datafolder_path(data_folder_path)

        # Start pipeline
        pipeline.run()

        if KeyboardInterrupt:
            pass                

        thread = class_saver.get_thread()
        if thread is not None:
            thread.join()
