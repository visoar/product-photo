# @title Image Alpha Blur
from PIL import Image, ImageFilter


def image_alpha_blur(image: Image):
    alpha = image.split()[-1]
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=2))

    def formula(al: int):
        if al < 128:
            return 0
        elif al < 255:
            return al - 85
        else:
            return 255

    alpha_data = [formula(a) for a in alpha.getdata()]
    alpha = Image.new('L', alpha.size)
    alpha.putdata(alpha_data)
    image.putalpha(alpha)
    return image
