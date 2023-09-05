# @title Image Enhance
import os

import cv2
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer


def get_up_sampler():
    # prepare up scale model
    model_path = '/RealESRGAN_x4plus.pth'
    if not os.path.isfile(model_path):
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        model_path = load_file_from_url(url=file_url, model_dir='/', progress=True)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=None,
        model=model,
        half=True)


def image_enhance(up_sampler, image: Image, out_scale=2):
    img_path = "enhance.png"
    image.save(img_path)
    output, _ = up_sampler.enhance(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), outscale=out_scale)
    cv2.imwrite(img_path, output)
    return Image.open(img_path)
