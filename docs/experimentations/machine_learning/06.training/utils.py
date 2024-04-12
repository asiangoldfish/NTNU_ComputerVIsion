import xml.etree.ElementTree as ET

def parse_annotations(annotations_path, skip_frames):
    """
    Parse annotations from an XML file.

    Args:
        annotations_path (str): The path to the XML file containing annotations.
        skip_frames (int): The number of frames to skip.

    Returns:
        list: A list of tuples containing the source, frame number, and bounding box coordinates.
    """

    tree = ET.parse(annotations_path)
    root = tree.getroot()
    annotations = list()    # A new `annotations` tuple for the current xml file
    source = root.find('meta').find('source').text

    for track in root.findall('track'):
        for idx, box in enumerate(track):
            if idx % skip_frames != 0:    # if the annotation doesn't match the given interval
                continue                    # skip the frame
            frame_nr = int(box.get('frame'))
            xtl = int(float(box.get('xtl')))
            ytl = int(float(box.get('ytl')))
            xbr = int(float(box.get('xbr')))
            ybr = int(float(box.get('ybr')))
            annotations.append({
                "source": source,
                "frame_num": frame_nr,
                "xtl": xtl,
                "ytl": ytl,
                "xbr": xbr, 
                "ybr": ybr
            })
    return annotations
