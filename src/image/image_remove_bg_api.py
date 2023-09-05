# @title Image Remove BG API
import json
from io import BytesIO

import requests
from PIL import Image


def image_remove_bg_api(image: Image, api_key: str):
    image.save("./segmentation.png")
    response = requests.request(
        "POST",
        "https://techsz.aoscdn.com/api/tasks/visual/segmentation",
        headers={'X-API-KEY': api_key},
        data={'sync': '1'},
        files={'image_file': open('./segmentation.png', 'rb')}
    )
    data = json.loads(response.text)
    image_url = data['data']['image']
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))
