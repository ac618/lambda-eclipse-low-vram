import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer
)

from diffusers import DiffusionPipeline
from utils import get_image_grid, linear_interpolation, slerp

from src.pipelines.pipeline_kandinsky_subject_prior import KandinskyPriorPipeline
from src.priors.lambda_prior_transformer import PriorTransformer


def main(concept1a, concept1b, concept2a, concept2b, prompt, seed=19645, interpolation_steps=4):
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        projection_dim=1280,
        torch_dtype=torch.float32,
    )
    tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

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
    pipe = pipe.to("cuda")

    raw_data = {
        "prompt": prompt,
        "subject_images": [concept1a["image_path"], concept1b["image_path"]],
        "subject_keywords": [concept1a["name"], concept1b["name"]]
    }
    start_top_left, negative_image_emb = pipe_prior(
        raw_data=raw_data#, generator=torch.Generator(device="cuda").manual_seed(seed)
    ).to_tuple()

    raw_data = {
        "prompt": prompt,
        "subject_images": [concept1a["image_path"], concept2b["image_path"]],
        "subject_keywords": [concept1a["name"], concept2b["name"]]
    }
    start_top_right, negative_image_emb = pipe_prior(
        raw_data=raw_data#, generator=torch.Generator(device="cuda").manual_seed(seed)
    ).to_tuple()

    raw_data = {
        "prompt": prompt,
        "subject_images": [concept2a["image_path"], concept1b["image_path"]],
        "subject_keywords": [concept2a["name"], concept1b["name"]]
    }
    start_bottom_left, negative_image_emb = pipe_prior(
        raw_data=raw_data#, generator=torch.Generator(device="cuda").manual_seed(seed)
    ).to_tuple()

    raw_data = {
        "prompt": prompt,
        "subject_images": [concept2a["image_path"], concept2b["image_path"]],
        "subject_keywords": [concept2a["name"], concept2b["name"]]
    }
    start_bottom_right, negative_image_emb = pipe_prior(
        raw_data=raw_data#, generator=torch.Generator(device="cuda").manual_seed(seed)
    ).to_tuple()
    
    
    gen_images = []

    for en in range(interpolation_steps + 1):
        top_left = linear_interpolation(
            start_top_left, start_bottom_left, t=en / interpolation_steps
        )
        top_right = linear_interpolation(
            start_top_right, start_bottom_right, t=en / interpolation_steps
        )
        for kk in range(interpolation_steps + 1):
            new_image_embeds = linear_interpolation(
                top_left, top_right, t=kk / interpolation_steps
            )

            images = pipe(
                num_inference_steps=50,
                image_embeds=new_image_embeds,
                negative_image_embeds=negative_image_emb,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images
            gen_images.append(images[0])

    return get_image_grid(
        gen_images, rows=interpolation_steps + 1, cols=interpolation_steps + 1
    )



if __name__ == "__main__":
    concept1a = {"image_path": "./assets/dog1.png", "name": "dog"}
    concept1b = {"image_path": "./assets/hat1.png", "name": "hat"}
    concept2a = {"image_path": "./assets/dog2.png", "name": "dog"}
    concept2b = {"image_path": "./assets/hat2.png", "name": "hat"}
    prompt = "a dog is wearing a hat"
    seed = 79846512
    interpolation_steps = 4
    
    without_interpolation_image = main(concept1a, concept1b, concept2a, concept2b, prompt, seed, interpolation_steps=1)
    without_interpolation_image.save("./assets/interpolation_without.png")

    without_interpolation_image = main(concept1a, concept1b, concept2a, concept2b, prompt, seed, interpolation_steps)
    without_interpolation_image.save("./assets/interpolation_with.png")