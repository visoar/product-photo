# @title Image Remove Background for cloth
from PIL import Image
from rembg import remove, new_session


def get_u2net_cloth_session():
    return new_session('u2net_cloth_seg')


def remove_background_for_cloth(u2net_cloth_session, image: Image):
    seg_image = remove(image, session=u2net_cloth_session)
    output_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    output_image.alpha_composite(seg_image, (0, 0), (0, 0))
    output_image.alpha_composite(seg_image, (0, 0), (0, image.size[1]))
    output_image.alpha_composite(seg_image, (0, 0), (0, image.size[1] * 2))
    return output_image
