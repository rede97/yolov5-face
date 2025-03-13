import os
import cv2
import onnx
import copy
import numpy as np
from onnxruntime.quantization import CalibrationDataReader

from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.config.custom_config import get_default_config
from quark.onnx import ModelQuantizer


def scale_image(img, max_border=640):
    shape = img.shape[:2]  # current shape [height, width]
    # Scale ratio (new / old)
    r = min(1.0, max_border / max(shape[0], shape[1]))
    new_unpad = min(640, int(round(shape[1] * r))), min(640, int(round(shape[0] * r)))
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    return img


def padding_image(img, new_shape=(640, 640)):
    h0, w0 = img.shape[:2]
    assert max(h0, w0) <= 640
    h_pad = (640 - h0) / 2
    w_pad = (640 - w0) / 2
    top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
    left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT
    )  # add border


def get_model_input_name(input_model_path: str) -> str:
    model = onnx.load(input_model_path)
    model_input_name = model.graph.input[0].name
    return model_input_name


class ImageDataReader(CalibrationDataReader):

    def __init__(self, calibration_image_folder: str, input_name: str, range=-1):
        self.enum_data = None

        self.input_name = input_name
        file_list = [
            f
            for f in os.listdir(calibration_image_folder)
            if f.endswith(".png") or f.endswith(".jpg")
        ]
        self.image_list = [
            os.path.join(calibration_image_folder, f) for f in file_list
        ][: min(range, len(file_list))]

    def _preprocess_images(self, image_path: str):
        print("preprocess:", image_path)
        input_image = cv2.imread(image_path)
        input_image = padding_image(scale_image(input_image))
        # Resize the input image. Because the size of Resnet50 is 224.
        input_data = np.array(input_image).astype(np.float32)
        # Custom Pre-Process
        input_data = input_data.transpose(2, 0, 1)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = input_data / 255.0
        return input_data

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [
                    {self.input_name: self._preprocess_images(image_path)}
                    for image_path in self.image_list
                ]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


# Set up quantization with a specified configuration
# For example, use "XINT8" for Ryzen AI INT8 quantization
xint8_config = get_default_config("XINT8")
quantization_config = Config(global_quant_config=xint8_config)


input_model_path = "weights/yolov5n-face-relu.onnx"
quantized_model_path = "weights/yolov5n-face-reluquantized.onnx"
calib_data_path = "data/widerface/val/images"
model_input_name = get_model_input_name(input_model_path)
calib_data_reader = ImageDataReader(calib_data_path, model_input_name, 50)
quantizer = ModelQuantizer(quantization_config)
quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)
