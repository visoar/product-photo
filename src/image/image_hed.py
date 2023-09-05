# @title Image Hed
from PIL import Image
from controlnet_aux import HEDdetector


def get_hed_detector():
    return HEDdetector.from_pretrained("lllyasviel/ControlNet")


def image_hed(hed_detector, image: Image):
    return hed_detector(image)
