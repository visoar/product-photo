# @title Image Mask
from PIL import Image


def image_mask(image: Image, reverse=False):
    def custom_formula(pixel):
        r, g, b, a = pixel
        if reverse:
            if a > 0:
                return 255, 255, 255, 255
            else:
                return 0, 0, 0, 255
        else:
            if a > 0:
                return 0, 0, 0, a
            else:
                return 255, 255, 255, 255

    mask_pixels = [custom_formula(pixel) for pixel in image.getdata()]
    mask_img = Image.new('RGBA', image.size)
    mask_img.putdata(mask_pixels)
    return mask_img
