# @title Image Merge PNG
from PIL import Image


def image_merge_png(bottom_img: Image, top_img: Image, ignore_color=None):
    if ignore_color is None:
        if bottom_img.mode != 'RGBA':
            bottom_img = bottom_img.convert('RGBA')
        bottom_img.alpha_composite(top_img)
        return bottom_img

    bottom_pixels = [pixel for pixel in bottom_img.getdata()]
    top_pixels = [pixel for pixel in top_img.getdata()]

    def pixel_formula(i):
        top_pixel = top_pixels[i]
        bottom_pixel = bottom_pixels[i]
        alpha = top_pixel[3]
        if ignore_color and top_pixel == ignore_color:
            return bottom_pixel
        if alpha > 0:
            return (
                int(((255 - alpha) * bottom_pixel[0] + alpha * top_pixel[0]) / 255),
                int(((255 - alpha) * bottom_pixel[1] + alpha * top_pixel[1]) / 255),
                int(((255 - alpha) * bottom_pixel[2] + alpha * top_pixel[2]) / 255),
                255
            )
        else:
            return bottom_pixel

    new_ima_pixels = [pixel_formula(i) for i in range(len(bottom_pixels))]
    new_img = Image.new('RGBA', bottom_img.size)
    new_img.putdata(new_ima_pixels)
    return new_img
