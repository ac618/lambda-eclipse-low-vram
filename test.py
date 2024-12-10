import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from src.pipelines.pipeline_kandinsky_subject_prior import KandinskyPriorPipeline
from src.priors.lambda_prior_transformer import PriorTransformer
from diffusers import DiffusionPipeline

from PIL import Image
import numpy as np
import os

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

    args = parser.parse_args()
    return args

def main(args):
    ####################################
    # Get the text and vision encoders #
    ####################################
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        projection_dim=1280,
        torch_dtype=torch.float32,
    )

    tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    vision_encoder = (
        CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            projection_dim=1280,
        )
        .eval()
        .requires_grad_(False)
    ).to("cuda")

    image_processor = CLIPImageProcessor.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        projection_dim=1280,
        do_rescale=True,
    )


    ######################
    # Load the pipelines #
    ######################
    prior = PriorTransformer.from_pretrained("ECLIPSE-Community/Lambda-ECLIPSE-Prior-v1.0")
    pipe_prior = KandinskyPriorPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-prior",
        prior=prior,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    ).to("cuda")

    pipe = DiffusionPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder"
    ).to("cuda")


    ###################################################
    # Get zero-embedding for unconditional generation #
    ###################################################
    zero_img = torch.zeros(
        1,
        3,
        vision_encoder.config.image_size,
        vision_encoder.config.image_size,
    ).to(device="cuda", dtype=vision_encoder.dtype)

    control_embedding = vision_encoder(zero_img).image_embeds


    #########################
    # Setup the base prompt #
    #########################
    prompt = args.prompt
    txt = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    txt_items = {k: v.to("cuda") for k, v in txt.items()}
    new_feats = text_encoder(**txt_items)
    new_last_hidden_states = new_feats.last_hidden_state[0].detach().cpu().numpy()


    ########################################################
    # Get the embedding corresponding to the first subject #
    ########################################################
    img = Image.open(args.subject1_path)
    mask_img = image_processor(img, return_tensors="pt").to("cuda")
    vision_feats = vision_encoder(**mask_img).image_embeds

    entity_tokens = tokenizer(args.subject1_name)["input_ids"][1:-1]

    found = True
    for tid in entity_tokens:
        indices = np.where(txt_items["input_ids"][0].cpu().numpy() == tid)[0]
        if len(indices)==0:
            found = False
        new_last_hidden_states[indices] = vision_feats[0].cpu().numpy()
    if found==False:
        print("There is some mismatch issues with subject1 name and prompt. Results might get affected.")

    if args.subject2_path and args.subject2_name:
        #########################################################
        # Get the embedding corresponding to the second subject #
        #########################################################
        img = Image.open(args.subject2_path)
        mask_img = image_processor(img, return_tensors="pt").to("cuda")
        vision_feats = vision_encoder(**mask_img).image_embeds

        entity_tokens = tokenizer(args.subject2_name)["input_ids"][1:-1]

        found = True
        for tid in entity_tokens:
            indices = np.where(txt_items["input_ids"][0].cpu().numpy() == tid)[0]
            if len(indices)==0:
                found = False
            new_last_hidden_states[indices] = vision_feats[0].cpu().numpy()
        if found==False:
            print("There is some mismatch issues with subject2 name and prompt. Results might get affected.")
    else:
        print("Second concept is not provided!")

    ####################################
    # Compile and generate final image #
    ####################################
    text_feats = {
        "prompt_embeds": new_feats.text_embeds.to("cuda"),
        "text_encoder_hidden_states": torch.tensor(new_last_hidden_states)
        .unsqueeze(0)
        .to("cuda"),
        "text_mask": txt_items["attention_mask"].to("cuda"),
    }

    image_emb, negative_image_emb = pipe_prior(
        text_feats=text_feats,
        control_embedding=control_embedding,
    ).to_tuple()
    image = pipe(
        image_embeds=image_emb,
        negative_image_embeds=negative_image_emb,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images

    image[0].save(os.path.join(args.output_dir, f'{prompt.replace(" ", "_")}.png'))


if __name__ == "__main__":
    args = get_parser()
    main(args)