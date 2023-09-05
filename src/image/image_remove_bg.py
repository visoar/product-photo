# @title Image Remove Background
import torch
from PIL import Image
from carvekit.ml.files.models_loc import tracer_b7_pretrained
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface


def get_tracer_b7_interface():
    tracer_b7_pretrained()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return init_interface(MLConfig(segmentation_network="tracer_b7",
                                   preprocessing_method="none",
                                   postprocessing_method="fba",
                                   seg_mask_size=640,
                                   trimap_dilation=30,
                                   trimap_erosion=5,
                                   device=device))


def image_remove_bg(tracer_b7_interface,
                    image: Image):
    if image.mode == 'RGBA':
        transparency_count = 0
        total_count = 0
        for pixel in image.getdata():
            total_count += 1
            if pixel[3] == 0:
                transparency_count += 1
        # 透明面积超过1/3, 默认已抠图
        if transparency_count > total_count / 3:
            return image

    return tracer_b7_interface([image])[0]
