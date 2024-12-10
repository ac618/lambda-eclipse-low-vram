import torch
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image

class DreamBoothDataset(torch.utils.data.Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        subject_data_root,
        instance_prompt,
        tokenizer=None,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        # add the image processor and encoder to prepare the image embeddings
        image_processor=None,
        image_encoder=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.subject_data_root = Path(subject_data_root)
        if not self.subject_data_root.exists():
            raise ValueError("Segmented images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.subject_images_path = list(Path(subject_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.num_subject_images = len(self.subject_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        # precompute the image embeddings to save VRAM
        self.image_embeds = None
        if image_processor is not None and image_encoder is not None:
            self.image_embeds = []
            for en_i in range(len(self.instance_images_path)):
                clip_images = image_processor(Image.open(self.instance_images_path[en_i]), return_tensors="pt").to(image_encoder.device)
                self.image_embeds.append(image_encoder(**clip_images).image_embeds[0])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = self.image_transforms(instance_image)
        instance_image = np.array(instance_image).astype(np.float32) / 127.5 - 1
        instance_image = torch.from_numpy(np.transpose(instance_image, [2, 0, 1]))
    
        subject_image = Image.open(self.subject_images_path[index % self.num_subject_images])
        subject_image = self.image_transforms(subject_image)
        subject_image = np.array(subject_image).astype(np.float32) / 127.5 - 1
        subject_image = torch.from_numpy(np.transpose(subject_image, [2, 0, 1]))

        # if not instance_image.mode == "RGB":
        #     instance_image = instance_image.convert("RGB")
        # example["instance_pil_images"] = instance_image
        example["instance_images"] = instance_image
        example["instance_pil_images"] = [str(self.instance_images_path[index % self.num_instance_images])]
        example["subject_pil_images"] = [str(self.subject_images_path[index % self.num_subject_images])]
        example["instance_prompt"] = self.instance_prompt
        # get the image embeddings
        example["image_embeds"] = self.image_embeds[index % self.num_instance_images] if self.image_embeds is not None else None
        return example

if __name__ == "__main__":
    train_dataset = DreamBoothDataset(
        instance_data_root="concept_image/cat2",
        subject_data_root="concept_image/cat2",
        instance_prompt="A cat2",
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        # tokenizer=tokenizer,
        size=512,
        center_crop=True,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        # tokenizer_max_length=args.tokenizer_max_length,
        image_processor=None,
        image_encoder=None,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # collate_fn=collate_fn,
        batch_size=2,
        num_workers=4,
    )
    # sample a batch
    for batch in train_dataloader:
        print(batch["instance_pil_images"])
        print(len(batch["instance_pil_images"]))
        break