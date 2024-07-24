from PIL import Image
import roboflow
import tempfile
import io

class RoboflowUploader:
    # Thin wrapper over Roboflow's upload API

    def __init__(self, workspace_name: str, dataset_name: str, api_key: str) -> "RoboflowUploader":
        rf = roboflow.Roboflow(api_key=api_key)
        self.project = rf.workspace(workspace_name).project(dataset_name)

    def upload(self, frame, class_names, bboxes):
        # Save the image to a bytes buffer
        img = Image.fromarray(frame)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        # Roboflow requires file paths
        img_file = tempfile.NamedTemporaryFile(suffix=".jpg")
        img.save(img_file, format="JPEG")
        img_file.flush()

        annotation = make_voc_annotations(class_names, bboxes, frame.shape[1], frame.shape[0])
        ann_file = tempfile.NamedTemporaryFile(suffix=".xml", delete=False)
        with open(ann_file.name, "w") as f:
            f.write(annotation)

        self.project.upload(
            image_path=img_file.name,
            annotation_path=ann_file.name
        )


def make_voc_annotations(cls_names, bboxes, width, height):
    HEADER = f"""
    <annotation>
        <source>
            <database>auto-generated from OAK camera</database>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>3</depth>
        </size>
    """

    FOOTER = """
    </annotation>
    """

    voc_xml_str = ""
    voc_xml_str += HEADER

    for cls_name, bbox in zip(cls_names, bboxes):
        voc_xml_str += make_obj_xml_string(cls_name, *bbox)

    voc_xml_str += FOOTER

    return voc_xml_str

def make_obj_xml_string(cls_name: str, xmin: int, ymin: int, xmax: int, ymax: int) -> str:
    xml_str = f"""
	<object>
		<name>{cls_name}</name>
		<bndbox>
			<xmin>{xmin}</xmin>
			<xmax>{xmax}</xmax>
			<ymin>{ymin}</ymin>
			<ymax>{ymax}</ymax>
		</bndbox>
	</object>
    """

    return xml_str
