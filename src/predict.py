# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import math
import os
from typing import List

import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from huggingface_hub import login

from gen_prompt import gen_prompt
from image.image_alpha_blur import image_alpha_blur
from image.image_depth import image_depth, get_depth_estimator
from image.image_enhance import image_enhance, get_up_sampler
from image.image_hed import image_hed, get_hed_detector
from image.image_mask import image_mask
from image.image_matrix import image_matrix
from image.image_merge_png import image_merge_png
from image.image_remove_bg import image_remove_bg, get_tracer_b7_interface
from stable_diffusion import StableDiffusion


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        login("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")
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        # self.sd_hed = StableDiffusion(
        #     inpaint_model="saik0s/realistic_vision_inpainting",
        #     controlnet_model="lllyasviel/sd-controlnet-hed"
        # )

        # self.sd_hed = StableDiffusion(
        #     inpaint_model="stabilityai/stable-diffusion-2-inpainting",
        #     controlnet_model="thibaud/controlnet-sd21-hed-diffusers"
        # )

        # self.sd_depth = StableDiffusion(
        #     inpaint_model="saik0s/realistic_vision_inpainting",
        #     controlnet_model="lllyasviel/sd-controlnet-depth"
        # )

        # self.sd_depth = StableDiffusion(
        #     inpaint_model="stabilityai/stable-diffusion-2-inpainting",
        #     controlnet_model="thibaud/controlnet-sd21-depth-diffusers"
        # )

        self.sd_control = StableDiffusion(
            inpaint_model="stabilityai/stable-diffusion-2-inpainting",
            controlnet_model=["thibaud/controlnet-sd21-hed-diffusers",
                              "thibaud/controlnet-sd21-depth-diffusers"]
        )

        self.depth_estimator = get_depth_estimator()
        self.up_sampler = get_up_sampler()
        self.hed_detector = get_hed_detector()
        self.tracer_b7_interface = get_tracer_b7_interface()

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
            self,
            image_path: Path = Input(description="input image"),
            product_size: str = Input(
                description="Max product size",
                choices=["Original", "0.6 * width", "0.5 * width", "0.4 * width", "0.3 * width", "0.2 * width"],
                default='0.6 * width'
            ),
            prompt: str = Input(
                description="Product name or prompt"
            ),
            negative_prompt: str = Input(
                description="Anything you don't want in the photo",
                default="low quality, out of frame, illustration, 3d, sepia, painting, cartoons, sketch, watermark, text, Logo, advertisement"
            ),
            api_key: str = Input(
                description="OpenAI api_key, enhance prompt with ChatGPT if provided",
                default=None
            ),
            pixel: str = Input(
                description="image total pixel",
                choices=['512 * 512', '768 * 768', '1024 * 1024'],
                default='512 * 512'
            ),
            scale: int = Input(
                description="Factor to scale image by (maximum: 4)",
                default=3,
                le=4,
                ge=0
            ),
            image_num: int = Input(
                description="Number of image to generate",
                default=1,
                le=4,
                ge=0
            ),
            guidance_scale: float = Input(
                description="Guidance Scale",
                default=7.5
            ),
            num_inference_steps: int = Input(
                description="Inference Steps",
                default=20
            ),
            manual_seed: int = Input(
                description="Seed",
                default=-1
            )

    ) -> List[Path]:
        """Run a single prediction on the model"""

        if image_num > 0 and api_key is not None and len(api_key) > 0:
            prepare_prompt_thread = gen_prompt(api_key, prompt)
            prepare_prompt_thread.start()
        else:
            prepare_prompt_thread = None

        if pixel == '512 * 512':
            pixel_size = 512
            out_scale = 3
        if pixel == '768 * 768':
            pixel_size = 768
            out_scale = 2
        if pixel == '1024 * 1024':
            pixel_size = 1024
            out_scale = 2

        image = Image.open(image_path).convert('RGBA')

        # 调整输入图片的大小
        image_width, image_height = image.size

        if image_width > image_height:
            limit_size = int(pixel_size * math.sqrt(image_width / image_height))
            limit_size = limit_size - limit_size % 8
            image_height = int(1.0 * image_height / image_width * limit_size * out_scale)
            image_height = int(image_height / out_scale - image_height / out_scale % 8) * out_scale
            image_width = limit_size * out_scale
        else:
            limit_size = int(pixel_size * math.sqrt(image_height / image_width))
            limit_size = limit_size - limit_size % 8
            image_width = int(1.0 * image_width / image_height * limit_size * out_scale)
            image_width = int(image_width / out_scale - image_width / out_scale % 8) * out_scale
            image_height = limit_size * out_scale
        image_width = image_width - image_width % 8
        image_height = image_height - image_height % 8

        image = image.resize((image_width, image_height))

        print('1. Image matting:', (image_width, image_height))
        top_img = image_remove_bg(self.tracer_b7_interface, image)

        if product_size == 'Original':
            limit_product = None
        elif product_size == '0.6 * width':
            limit_product = 0.6
        elif product_size == '0.5 * width':
            limit_product = 0.5
        elif product_size == '0.4 * width':
            limit_product = 0.4
        elif product_size == '0.3 * width':
            limit_product = 0.3
        elif product_size == '0.2 * width':
            limit_product = 0.2
        else:
            limit_product = None

        # 控制产品大小, 不超过黄金分割
        if limit_product is not None:
            matrix = image_matrix(top_img)
            crop_region = top_img.crop(matrix)
            crop_scale = min(top_img.size[0] * limit_product / crop_region.size[0],
                             top_img.size[1] * limit_product / crop_region.size[1])
            if crop_scale < 1:
                top_img = Image.new('RGBA', top_img.size)
                top_img.paste(
                    crop_region.resize((int(crop_region.size[0] * crop_scale), int(crop_region.size[1] * crop_scale))),
                    (matrix[0] + int(crop_region.size[0] * (1 - crop_scale) / 2),
                     matrix[1] + int(crop_region.size[1] * (1 - crop_scale) / 2))
                )

        top_img.save('top.png')
        if image_num == 0:
            return [Path("top.png")]

        inpaint_width = int(image_width / out_scale)
        inpaint_height = int(image_height / out_scale)

        # 计算图片的蒙版图像
        top_inpaint_mask_img = image_mask(top_img).resize(
            (inpaint_width, inpaint_height))

        if prepare_prompt_thread is not None:
            prompt = prepare_prompt_thread.join()

        sub_prompts = prompt.split(';')
        sub_prompts.append("photorealistic++, Breathtaking+")

        prompt = "(" + ",".join(['"' + x.strip() + '"' for x in sub_prompts]) + ").and()"
        negative_prompt_hed = "(" + ",".join(
            ['"reflection, distorted, ' + negative_prompt + '"' for x in sub_prompts]) + ").and()"
        negative_prompt = "(" + ",".join(['"distorted,' + negative_prompt + '"' for x in sub_prompts]) + ").and()"

        print('2. Generate ad images')
        inpaint_width = int(image_width / out_scale)
        inpaint_height = int(image_height / out_scale)
        if manual_seed == -1:
            seed = torch.seed()
        else:
            seed = manual_seed
        print('Seed:', seed)

        top_inpaint_img = top_img.resize((inpaint_width, inpaint_height))
        hed_control_image = image_hed(self.hed_detector, top_inpaint_img)

        layout_image_width = int(inpaint_width * 512 / pixel_size)
        layout_image_width = layout_image_width - layout_image_width % 8
        layout_inpaint_height = int(inpaint_height * 512 / pixel_size)
        layout_inpaint_height = layout_inpaint_height - layout_inpaint_height % 8
        layout_hed_control_image = hed_control_image.resize((layout_image_width, layout_inpaint_height))
        layout_inpaint_img = top_inpaint_img.resize((layout_image_width, layout_inpaint_height))
        layout_img = self.sd_control.inpaint(
            image=layout_inpaint_img,
            mask_image=top_inpaint_mask_img.resize((layout_image_width, layout_inpaint_height)),
            prompt=prompt,
            negative_prompt=negative_prompt_hed,
            height=layout_inpaint_height,
            width=layout_image_width,
            guidance_scale=guidance_scale,
            num_inference_steps=5,
            control_image=[layout_hed_control_image, layout_hed_control_image],
            controlnet_conditioning_scale=[0.6, 0.0],
            control_guidance_start=[0.0, 0.0],
            control_guidance_end=[1.0, 0.1],
            generator=torch.manual_seed(seed)
        )[0]

        layout_img = image_merge_png(layout_img, layout_inpaint_img)

        output_list = [Path("top.png")]
        # layout_img.convert('RGB').save('ad_inpaint.jpg')
        # output_list.append(Path('ad_inpaint.jpg'))

        depth_control_image = image_depth(self.depth_estimator, layout_img).resize((inpaint_width, inpaint_height))

        inpaint_images = self.sd_control.inpaint(
            image=top_inpaint_img,
            mask_image=top_inpaint_mask_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=inpaint_height,
            width=inpaint_width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=image_num,
            control_image=[hed_control_image, depth_control_image],
            controlnet_conditioning_scale=[0.2, 0.7],
            control_guidance_start=[0.0, 0.0],
            control_guidance_end=[1.0, 1.0],
            generator=torch.manual_seed(seed)
        )

        print('4. Super resolution image')
        for index in range(image_num):
            if scale > 1:
                inpaint_up_img = image_enhance(self.up_sampler, inpaint_images[index], scale)
                output1_img = image_merge_png(inpaint_up_img, image_alpha_blur(
                    top_img.resize((inpaint_width * scale, inpaint_height * scale))))
                output1_img.convert('RGB').save('ad_inpaint_' + str(index) + '.jpg')
            else:
                output1_img = image_merge_png(inpaint_images[index], image_alpha_blur(
                    top_img.resize((inpaint_width, inpaint_height))))
                output1_img.convert('RGB').save('ad_inpaint_' + str(index) + '.jpg')
            print(' - image ' + str(index))

        for index in range(image_num):
            output_list.append(Path('ad_inpaint_' + str(index) + '.jpg'))

        return output_list
