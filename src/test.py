# @title Test
from PIL import Image

from image.image_seg import image_seg, get_seg_segmentor, get_seg_processor

processor = get_seg_processor()
segmentor = get_seg_segmentor()

image = Image.open('sofa1.jpg')

out_image = image_seg(processor, segmentor, image)
out_image.save('out.png')
