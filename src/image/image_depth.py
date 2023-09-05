# @title Image Depth
import numpy as np
from PIL import Image
from transformers import pipeline


def get_depth_estimator():
    return pipeline('depth-estimation')


def image_depth(depth_estimator, image: Image):
    depth_image = depth_estimator(image)['depth']
    depth_image = np.array(depth_image)
    depth_image = depth_image[:, :, None]
    depth_image = np.concatenate(3 * [depth_image], axis=2)
    depth_image = Image.fromarray(depth_image)
    return depth_image
