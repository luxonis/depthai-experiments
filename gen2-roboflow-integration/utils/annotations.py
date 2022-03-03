"""
Creates VOC-style annotation strings
"""


def make_voc_annotations(cls_names, bboxes, img_w=300, img_h=300):

    HEADER = f"""
    <annotation>
        <source>
            <database>auto-generated from OAK camera</database>
        </source>
        <size>
            <width>{img_w}</width>
            <height>{img_h}</height>
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


def make_obj_xml_string(
    cls_name: str, xmin: int, ymin: int, xmax: int, ymax: int
) -> str:
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


if __name__ == "__main__":

    # Test code

    anno = make_voc_annotations(
        ["helmet", "helmet"], [[179, 85, 231, 144], [112, 145, 135, 175]]
    )

    print(anno)
