# @title Image Openpose detector
from PIL import Image
from controlnet_aux import OpenposeDetector


def get_openpose_detector():
    return OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


def image_openpose(openpose_detector, image: Image):
    return openpose_detector(image, hand_and_face=True)
