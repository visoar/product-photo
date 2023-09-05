import torch
from basicsr.utils.download_util import load_file_from_url
from carvekit.ml.files.models_loc import tracer_b7_pretrained
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
from controlnet_aux import HEDdetector
from diffusers import ControlNetModel, StableDiffusionInpaintPipeline
from transformers import pipeline

tracer_b7_pretrained()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
init_interface(
    MLConfig(segmentation_network='tracer_b7',
             preprocessing_method='none',
             postprocessing_method='fba',
             seg_mask_size=640,
             trimap_dilation=30,
             trimap_erosion=5,
             device=device)
)

load_file_from_url(url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                   model_dir='./', progress=True)

HEDdetector.from_pretrained('lllyasviel/ControlNet')
pipeline('depth-estimation')

ControlNetModel.from_pretrained('thibaud/controlnet-sd21-hed-diffusers', torch_dtype=torch.float16)

ControlNetModel.from_pretrained('thibaud/controlnet-sd21-depth-diffusers', torch_dtype=torch.float16)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'stabilityai/stable-diffusion-2-inpainting', torch_dtype=torch.float16
)
