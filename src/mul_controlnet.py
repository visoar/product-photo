# @title Inpaint Pipeline
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


def make_inpaint_condition(img, img_mask):
    img = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    img_mask = np.array(img_mask.convert("L"))
    assert img.shape[0:1] == img_mask.shape[0:1], "image and image_mask must have the same image size"
    img[img_mask < 128] = -1.0  # set as masked pixel
    img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)
    img = torch.from_numpy(img)
    return img


controlnet_softedge = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge',
                                                      torch_dtype=torch.float16)
controlnet_depth = ControlNetModel.from_pretrained('lllyasviel/control_v11f1p_sd15_depth', torch_dtype=torch.float16)
controlnet_inpaint = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_inpaint', torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=[controlnet_depth, controlnet_softedge, controlnet_inpaint],
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.enable_model_cpu_offload()
pipe.to('cuda')
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()
