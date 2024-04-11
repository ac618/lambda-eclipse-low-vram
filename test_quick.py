import os
import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPImageProcessor
)
from src.pipelines.pipeline_kandinsky_subject_prior import KandinskyPriorPipeline
from src.priors.lambda_prior_transformer import PriorTransformer
from diffusers import DiffusionPipeline, UNet2DConditionModel

import cv2 as cv
from PIL import Image 

def get_canny_edge(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img, 100, 200)
    img = Image.fromarray(edges).convert("RGB")
    img.save("./assets/canny_gen.png")
    return img

# write the argument parser
def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tuning CLIP model on Kandinsky dataset")

    parser.add_argument("--prompt", type=str, required=True, help="Path to the checkpoint")
    parser.add_argument("--subject1_path", type=str, required=True, help="Batch size")
    parser.add_argument("--subject1_name", type=str, required=True, help="Learning rate")
    parser.add_argument("--subject2_path", type=str, default=None, help="Batch size")
    parser.add_argument("--subject2_name", type=str, default=None, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./assets/", help="Output directory")
    parser.add_argument("--unet_checkpoint", type=str, default=None, help="Finetuned UNet model FOLDER path")
    parser.add_argument("--canny_image", type=str, default=None, help="Path to reference image to extract Canny edge map")

    args = parser.parse_args()
    return args

def main(args):
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        projection_dim=1280,
        torch_dtype=torch.float32,
    )
    tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    image_processor = CLIPImageProcessor.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        do_rescale=True,  # subfolder="image_processor"
    )

    prior = PriorTransformer.from_pretrained("ECLIPSE-Community/Lambda-ECLIPSE-Prior-v1.0")
    pipe_prior = KandinskyPriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        prior=prior,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    ).to("cuda")

    pipe = DiffusionPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder"
    )
    if args.unet_checkpoint is not None:
        print('Loading UNet Checkpoint')
        if os.path.isfile(args.unet_checkpoint):
            unet = UNet2DConditionModel.from_pretrained(args.unet_checkpoint)
        else:
            unet = UNet2DConditionModel.from_pretrained(args.unet_checkpoint, subfolder="unet")
        pipe.unet = unet

    pipe = pipe.to("cuda")

    raw_data = {
        "prompt": args.prompt,
        "subject_images": [args.subject1_path],
        "subject_keywords": [args.subject1_name]
    }
    if args.subject2_path is not None:
        raw_data["subject_images"].append(args.subject2_path)
        raw_data["subject_keywords"].append(args.subject2_name)

    canny_image_emb = None
    if args.canny_image:
        canny_img = torch.tensor(image_processor(get_canny_edge(args.canny_image)).pixel_values[0]).unsqueeze(0).to(pipe_prior.device)
        canny_image_emb = pipe_prior.image_encoder(canny_img).image_embeds
        print("canny image considered")

    image_emb, negative_image_emb = pipe_prior(
        raw_data=raw_data,
        control_embedding=canny_image_emb, 
    ).to_tuple()

    image = pipe(
        image_embeds=image_emb,
        negative_image_embeds=negative_image_emb,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images

    image[0].save(os.path.join(args.output_dir, f'{args.prompt.replace(" ", "_")}.png'))


if __name__ == "__main__":
    args = get_parser()
    main(args)