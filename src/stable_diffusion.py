# @title StableDiffusion
import torch
from PIL import Image
from compel import Compel
from diffusers import ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetInpaintPipeline


class StableDiffusion:
    def __init__(self, inpaint_model, controlnet_model):
        if not isinstance(controlnet_model, list):
            controlnet_model = [controlnet_model]
        controlnet = list(
            map(lambda model: ControlNetModel.from_pretrained(model, torch_dtype=torch.float16), controlnet_model)
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            inpaint_model, controlnet=controlnet, torch_dtype=torch.float16
        )

        # speed up diffusion process with faster scheduler and memory optimization
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # remove following line if xformers is not installed
        pipe.to('cuda')
        pipe.enable_xformers_memory_efficient_attention()
        # pipe.enable_attention_slicing()

        self.inpaint_pipe = pipe
        self.compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    def inpaint(self, prompt, negative_prompt, **args) -> list[Image]:
        with torch.no_grad():
            return self.inpaint_pipe(
                prompt_embeds=self.compel_proc(prompt),
                negative_prompt_embeds=self.compel_proc(negative_prompt),
                **args
            ).images
