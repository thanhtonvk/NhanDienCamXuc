import numpy as np
import cv2
from modules.SCRFD import SCRFD
from skimage import transform


def resizeBox(box):
    x_min,y_min,x_max,y_max  = box
    width = x_max - x_min
    height = y_max - y_min

    side_length = max(width, height)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    new_x_min = x_center - side_length / 2
    new_y_min = y_center - side_length / 2
    new_x_max = x_center + side_length / 2
    new_y_max = y_center + side_length / 2
    return new_x_min,new_y_min,new_x_max,new_y_max




class FaceDetector:
    def __init__(self, ctx_id=0, det_size=(640, 640)):
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.model = SCRFD(model_file="models/scr_face_detector.onnx")
        self.model.prepare()

    def detect(
            self,
            np_image: np.ndarray,
            confidence_threshold=0.5,
    ):
        bboxes = []
        predictions = self.model.get(
            np_image, threshold=confidence_threshold, input_size=self.det_size)
        if len(predictions) != 0:
            for _, face in enumerate(predictions):
                bbox = face["bbox"]
                bbox = resizeBox(bbox)
                bbox = list(map(int, bbox))
                bboxes.append(bbox)
        return bboxes
