from __future__ import annotations
import pathlib
import gradio as gr
import torch
import os
import PIL
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Any

from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    CLIPTokenizer
)

from transformers import CLIPTokenizer
from src.priors.lambda_prior_transformer import (
    PriorTransformer,
)  # original huggingface prior transformer without time conditioning
from src.pipelines.pipeline_kandinsky_subject_prior import KandinskyPriorPipeline

from diffusers import DiffusionPipeline
from PIL import Image

class Model:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.text_encoder = (
            CLIPTextModelWithProjection.from_pretrained(
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                projection_dim=1280,
                torch_dtype=torch.float16,
            )
            .eval()
            .requires_grad_(False)
        ).to("cuda")

        self.tokenizer = CLIPTokenizer.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        )

        prior = PriorTransformer.from_pretrained(
            "ECLIPSE-Community/Lambda-ECLIPSE-Prior-v1.0",
            torch_dtype=torch.float16,
        )

        self.pipe_prior = KandinskyPriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior",
            prior=prior,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.pipe = DiffusionPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        ).to(self.device)
        
    def inference(self, raw_data):
        image_emb, negative_image_emb = self.pipe_prior(
            raw_data=raw_data,
        ).to_tuple()
        image = self.pipe(
            image_embeds=image_emb,
            negative_image_embeds=negative_image_emb,
            num_inference_steps=50,
            guidance_scale=4.0,
        ).images[0]
        return image
    
    def process_data(self,
                     image: PIL.Image.Image,
                     keyword: str,
                     image2: PIL.Image.Image,
                     keyword2: str, 
                     text: str,
                     ) -> dict[str, Any]:
        print(f"keyword : {keyword}, keyword2 : {keyword2}, prompt : {text}")
        device = torch.device(self.device)
        data: dict[str, Any] = {}
        data['text'] = text
       
        txt = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        txt_items = {k: v.to(device) for k, v in txt.items()}
        new_feats = self.text_encoder(**txt_items)
        new_last_hidden_states = new_feats.last_hidden_state[0].cpu().numpy()
        
        plt.imshow(image)
        plt.title('image')
        plt.savefig('image_testt2.png')
        plt.show()
        
        mask_img = self.image_processor(image, return_tensors="pt").to("cuda")
        vision_feats = self.vision_encoder(
            **mask_img
        ).image_embeds
        
        entity_tokens = self.tokenizer(keyword)["input_ids"][1:-1]
        for tid in entity_tokens:
            indices = np.where(txt_items["input_ids"][0].cpu().numpy() == tid)[0]
            new_last_hidden_states[indices] = vision_feats[0].cpu().numpy()
            print(indices)
        
        if image2 is not None:
            mask_img2 = self.image_processor(image2, return_tensors="pt").to("cuda")    
            vision_feats2 = self.vision_encoder(
                **mask_img2
            ).image_embeds
            if keyword2 is not None:
                entity_tokens = self.tokenizer(keyword2)["input_ids"][1:-1]
                for tid in entity_tokens:
                    indices = np.where(txt_items["input_ids"][0].cpu().numpy() == tid)[0]
                    new_last_hidden_states[indices] = vision_feats2[0].cpu().numpy()
                    print(indices)
        
        text_feats = {
            "prompt_embeds": new_feats.text_embeds.to("cuda"),
            "text_encoder_hidden_states": torch.tensor(new_last_hidden_states).unsqueeze(0).to("cuda"),
            "text_mask": txt_items["attention_mask"].to("cuda"),
        }      
        return text_feats

    def run(self,
            image: dict[str, PIL.Image.Image],
            keyword: str,
            image2: dict[str, PIL.Image.Image],
            keyword2: str,
            text: str,
            ):

        # aug_feats = self.process_data(image["composite"], keyword, image2["composite"], keyword2, text)
        sub_imgs = [image["composite"]]
        if image2:
            sub_imgs.append(image2["composite"])
        sun_keywords = [keyword]
        if keyword2:
            sun_keywords.append(keyword2)
        raw_data = {
            "prompt": text,
            "subject_images": sub_imgs,
            "subject_keywords": sun_keywords
        }
        image = self.inference(raw_data)
        return image

def create_demo():
    TITLE = '# [λ-Eclipse Demo](https://eclipse-t2i.github.io/Lambda-ECLIPSE/)'

    USAGE = '''To run the demo, you should:   
    1. Upload your image.   
    2. <span style='color: red;'>**Upload a masked subject image with white blankspace or whiten out manually using brush tool.**
    3. Input a Keyword i.e. 'Dog'
    4. For MultiSubject personalization,
     4-1. Upload another image.
     4-2. Input the Keyword i.e. 'Sunglasses'   
    3. Input proper text prompts, such as "A photo of Dog" or "A Dog wearing sunglasses", Please use the same keyword in the prompt.   
    4. Click the Run button.
    '''

    model = Model()

    with gr.Blocks() as demo:
        gr.Markdown(TITLE)
        gr.Markdown(USAGE)
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown(
                        'Upload your first masked subject image or mask out marginal space')
                    image = gr.ImageEditor(label='Input', type='pil', brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"))
                    keyword = gr.Text(
                        label='Keyword',
                        placeholder='e.g. "Dog", "Goofie"',
                        info='Keyword for first subject')
                    gr.Markdown(
                        'For Multi-Subject generation : Upload your second masked subject image or mask out marginal space')
                    image2 = gr.ImageEditor(label='Input', type='pil', brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"))
                    keyword2= gr.Text(
                        label='Keyword',
                        placeholder='e.g. "Sunglasses", "Grand Canyon"',
                        info='Keyword for second subject')
                    prompt = gr.Text(
                        label='Prompt',
                        placeholder='e.g. "A photo of dog", "A dog wearing sunglasses"',
                        info='Keep the keywords used previously in the prompt')

                run_button = gr.Button('Run')

            with gr.Column():
                result = gr.Image(label='Result')

        inputs = [
            image,
            keyword,
            image2,
            keyword2,
            prompt,
        ]
        
        gr.Examples(
            examples=[[os.path.join(os.path.dirname(__file__), "./assets/cat.png"), "cat", os.path.join(os.path.dirname(__file__), "./assets/blue_sunglasses.png"), "glasses", "A cat wearing glasses on a snowy field"]],
            inputs = inputs,
            fn=model.run,
            outputs=result,
        )
        
        run_button.click(fn=model.run, inputs=inputs, outputs=result)
    return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.queue(api_open=False).launch(share=True)