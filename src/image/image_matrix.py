# @title Image Matrix
from PIL import Image


def image_matrix(image: Image):
    top = image.size
    bottom = (0, 0)
    width, height = image.size
    index = 0
    for pixel in image.getdata():
        w = index % width
        h = int(index / width)
        if pixel[3] != 0:
            top = min(top[0], w), min(top[1], h)
            bottom = max(bottom[0], w), max(bottom[1], h)
        index += 1
    return top[0], top[1], bottom[0], bottom[1]
