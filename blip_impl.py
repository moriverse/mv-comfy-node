import torch

from dataclasses import dataclass
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BlipForConditionalGeneration,
)
from typing import Optional

CAPTION_MODELS = {
    "blip-base": "Salesforce/blip-image-captioning-base",  # 990MB
    "blip-large": "Salesforce/blip-image-captioning-large",  # 1.9GB
    "blip2-2.7b": "Salesforce/blip2-opt-2.7b",  # 15.5GB
    "blip2-flan-t5-xl": "Salesforce/blip2-flan-t5-xl",  # 15.77GB
    "git-large-coco": "microsoft/git-large-coco",  # 1.58GB
}

CACHE_URL_BASE = "https://huggingface.co/pharmapsychotic/ci-preprocess/resolve/main/"


@dataclass
class Config:
    # models can optionally be passed in directly
    caption_model = None
    caption_processor = None

    # blip settings
    caption_max_length: int = 64
    caption_model_name: Optional[str] = (
        "blip-large"  # use a key from CAPTION_MODELS or None
    )
    caption_offload: bool = False
    cache_dir: Optional[str] = None

    # interrogator settings
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    quiet: bool = False  # when quiet progress bars are not shown


class Interrogator:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.caption_offloaded = True

        self.load_caption_model()

    def load_caption_model(self):
        if self.config.caption_model is None and self.config.caption_model_name:
            if not self.config.quiet:
                print(f"Loading caption model {self.config.caption_model_name}...")
                print(f"Cache_dir: {self.config.cache_dir}")

            model_path = CAPTION_MODELS[self.config.caption_model_name]
            if self.config.caption_model_name.startswith("git-"):
                caption_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    cache_dir=self.config.cache_dir,
                )
            elif self.config.caption_model_name.startswith("blip2-"):
                from transformers import Blip2ForConditionalGeneration

                caption_model = Blip2ForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=self.dtype, cache_dir=self.config.cache_dir
                )
            else:
                caption_model = BlipForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=self.dtype, cache_dir=self.config.cache_dir
                )

            print(f"Loaded caption_model of type {type(caption_model)}")
            self.caption_processor = AutoProcessor.from_pretrained(
                model_path, cache_dir=self.config.cache_dir
            )

            caption_model.eval()
            if not self.config.caption_offload:
                caption_model = caption_model.to(self.config.device)
            self.caption_model = caption_model
        else:
            self.caption_model = self.config.caption_model
            self.caption_processor = self.config.caption_processor

    def generate_caption(self, pil_image: Image, input_txt="") -> str:
        assert self.caption_model is not None, "No caption model loaded."
        self._prepare_caption()

        if input_txt:
            inputs = self.caption_processor(
                images=pil_image, text=input_txt, return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(
                self.device
            )

        if not self.config.caption_model_name.startswith("git-"):
            inputs = inputs.to(self.dtype)

        tokens = self.caption_model.generate(
            **inputs, max_new_tokens=self.config.caption_max_length + len(input_txt)
        )
        caption = self.caption_processor.batch_decode(tokens, skip_special_tokens=True)[
            0
        ].strip()

        print(f"Generated {type(self.caption_model)} caption:")
        print(caption)
        print("-------------------------------------")

        return caption

    def _prepare_caption(self):
        if self.caption_offloaded:
            self.caption_model = self.caption_model.to(self.device)
            self.caption_offloaded = False
