import base64
import cv2
import io
import numpy as np
import os
import random
import requests
import torch
import typing as t

from gfpgan import GFPGANer
from PIL import Image, ImageOps

# Import Comfy components.
import folder_paths

from .mediapipe_impl import MediapipeSegmenter


USER_AGENT = "Moriverse/Comfy"


def tensor2np(image):
    return np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)


def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def np2tensor(image):
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


class LoadImagesFromUrl:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_temp_directory()
        self.filename_prefix = "TempImageFromUrl"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "urls": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    RETURN_NAMES = ("images",)
    CATEGORY = "Moriverse/image"
    FUNCTION = "load_image"

    def load_image(self, urls=""):
        urls = urls.strip().split("\n")
        images = self.load_images_from_url(urls)

        if len(images) == 0:
            raise Exception("No image found.")

        previews = []
        np_images = []

        for image in images:
            # save image to temp folder
            previews.append(
                self.prepare_image_for_preview(
                    image,
                    self.output_dir,
                    self.filename_prefix,
                )
            )
            np_images.append(pil2tensor(image))

        return {
            "ui": {"images": previews},
            "result": (np_images,),
        }

    def prepare_image_for_preview(
        self,
        image: Image.Image,
        output_dir: str,
        prefix=None,
    ):
        if prefix is None:
            prefix = "preview_" + "".join(
                random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
            )

        # save image to temp folder
        (
            outdir,
            filename,
            counter,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(
            prefix,
            output_dir,
            image.width,
            image.height,
        )
        file = f"{filename}_{counter:05}_.png"
        image.save(os.path.join(outdir, file), format="PNG", compress_level=4)

        return {
            "filename": file,
            "subfolder": subfolder,
            "type": "temp",
        }

    def load_images_from_url(self, urls: t.List[str]):
        images = []

        for url in urls:
            if url.startswith("data:image/"):
                image = Image.open(io.BytesIO(base64.b64decode(url.split(",")[1])))

            elif url.startswith("file://"):
                url = url[7:]
                if not os.path.isfile(url):
                    raise Exception(f"File {url} does not exist")

                image = Image.open(url)

            elif url.startswith("http://") or url.startswith("https://"):
                response = requests.get(url, timeout=5)
                if response.status_code != 200:
                    raise Exception(response.text)

                image = Image.open(io.BytesIO(response.content))

            elif url.startswith("/view?"):
                from urllib.parse import parse_qs

                qs = parse_qs(url[6:])
                filename = qs.get("name", qs.get("filename", None))
                if filename is None:
                    raise Exception(f"Invalid url: {url}")

                filename = filename[0]
                subfolder = qs.get("subfolder", None)
                if subfolder is not None:
                    filename = os.path.join(subfolder[0], filename)

                dirtype = qs.get("type", ["input"])
                if dirtype[0] == "input":
                    url = os.path.join(folder_paths.get_input_directory(), filename)
                elif dirtype[0] == "output":
                    url = os.path.join(folder_paths.get_output_directory(), filename)
                elif dirtype[0] == "temp":
                    url = os.path.join(folder_paths.get_temp_directory(), filename)
                else:
                    raise Exception(f"Invalid url: {url}")

                image = Image.open(url)
            elif url == "":
                continue
            else:
                url = folder_paths.get_annotated_filepath(url)
                if not os.path.isfile(url):
                    raise Exception(f"Invalid url: {url}")

                image = Image.open(url)

            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            images.append(image)

        return images


def _get_largest_part(parts):
    max_part = None
    max_area = 0
    for part in parts:
        bbox = part.bbox
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area >= max_area:
            max_area = area
            max_part = part
    return max_part


class FaceDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bbox_detector": ("BBOX_DETECTOR",),
                "images": ("IMAGE",),
            },
            "optional": {
                "segm": ("BOOLEAN", {"default": False}),
                "segm_detector": ("SEGM_DETECTOR",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "size": ("INT", {"default": 640, "min": 0, "max": 1024, "step": 64}),
                "face_ratio": (
                    "FLOAT",
                    {"default": 0.45, "min": 0.1, "max": 2, "step": 0.01},
                ),
                "segm_dilation": (
                    "INT",
                    {"default": 10, "min": 0, "max": 200, "step": 10},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"
    CATEGORY = "Moriverse/ops"

    def doit(
        self,
        bbox_detector,
        images,
        segm=False,
        segm_detector=None,
        threshold=0.5,
        size=640,
        face_ratio=0.45,
        segm_dilation=10,
    ):
        result = []

        for image in images:
            segs = bbox_detector.detect(
                image.unsqueeze(0),
                threshold,
                0,  # Dilation.
                1.0,  # Crop factor,
                10,  # Drop size.
            )
            cropped = self._crop(image, segs, face_ratio, size)
            if cropped is None:
                print("No face bbox detected.")
                continue

            if segm and segm_detector is not None:
                segm_segs = segm_detector.detect(
                    cropped.unsqueeze(0), threshold, segm_dilation, 10.0, 10
                )
                if len(segm_segs[1]) == 0:
                    print("No face detected from user image.")
                    continue

                face = segm_segs[1][0]
                mask = torch.from_numpy(face.cropped_mask)
                if len(mask.shape) == 2:
                    pass

                elif len(mask.shape) == 3:
                    mask = mask[0]

                else:
                    print(f"Unsupported face mask dim. Shape: {mask.shape}")

                mask_w, mask_h = mask.shape
                image_w, image_h, _ = cropped.shape
                assert (
                    mask_w == image_w and mask_h == image_h
                ), f"Mask size {mask.shape} not match with image size {segm_segs[0]}."

                # Composite image and mask.
                mask = mask.unsqueeze(-1).expand(-1, -1, cropped.shape[2])
                cropped = cropped * mask

            result.append(cropped)

        if len(result) == 0:
            raise Exception("No valid user image.")

        result = torch.stack(result, dim=0)
        return (result,)

    def _crop(self, image, segs, face_ratio, size):
        h, w = segs[0]
        face = _get_largest_part(segs[1])

        if face is None:
            return None

        bbox = face.bbox
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]

        crop_l = int(max(face_h, face_w) / face_ratio / 2)
        cx = int((bbox[2] + bbox[0]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        crop_up = min(cy, crop_l)
        crop_bo = min(h - cy, crop_l)
        crop_le = min(cx, crop_l)
        crop_ri = min(w - cx, crop_l)

        x_pad = (crop_l - crop_le, crop_l - crop_ri)
        y_pad = (crop_l - crop_up, crop_l - crop_bo)

        image = image.numpy()
        image = image[cy - crop_up : cy + crop_bo, cx - crop_le : cx + crop_ri, :]
        image = np.pad(image, (y_pad, x_pad, (0, 0)), "constant")
        image = cv2.resize(image, (size, size))
        image = cv2.cvtColor(image[:, :, ::-1], cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)

        return image


class CropFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "detector": ("BBOX_DETECTOR",),
                "image": ("IMAGE",),
            },
            "optional": {
                "threshold": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "expand": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
                ),
                "size": ("INT", {"default": 640, "min": 0, "max": 1024, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE", "CROP_REGION", "SIZE")
    FUNCTION = "doit"
    CATEGORY = "Moriverse/ops"

    def doit(self, detector, image, threshold=0.8, size=640, expand=1.0):
        assert len(image) == 1, "Can only crop one image at a time."

        segs = detector.detect(
            image, threshold, 0, 1.0, 10, None  # Dilation.  # Crop ratio.  # Drop size.
        )

        image_h, image_w = segs[0]
        face = _get_largest_part(segs[1])

        assert face is not None, "No face detected."

        cropped = face.cropped_image[0]
        h, w, _ = cropped.shape
        short = min(h, w)
        long = max(h, w)
        padding = (long - short) // 2

        offset_x, offset_y, *_ = face.crop_region
        x_start = offset_x if w < h else offset_x + padding
        y_start = offset_y + padding if w < h else offset_y
        x_end = x_start + short
        y_end = y_start + short

        margin = (short * expand) // 2

        tl_margin = min(margin, min(x_start, y_start))
        br_margin = min(margin, min(image_w - x_end, image_h - y_end))
        margin = int(min(tl_margin, br_margin))

        x_start = x_start - margin
        y_start = y_start - margin
        x_end = x_end + margin
        y_end = y_end + margin

        orginal_size = short + margin * 2
        crop_region = [x_start, y_start, x_end, y_end]
        cropped = image[:, y_start:y_end, x_start:x_end, :]

        single_pil = tensor2pil(cropped)
        scaled_pil = single_pil.resize((size, size), resample=LANCZOS)
        cropped = pil2tensor(scaled_pil)

        return (cropped, crop_region, orginal_size)


LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


class PasteFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "ref_image": ("IMAGE",),
                "crop_region": ("CROP_REGION",),
                "size": ("SIZE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"
    CATEGORY = "Moriverse/ops"

    def doit(self, base_image, ref_image, crop_region, size):
        tl_x, tl_y, br_x, br_y = crop_region
        w = br_x - tl_x
        h = br_y - tl_y

        single_pil = tensor2pil(ref_image)
        scaled_pil = single_pil.resize((size, size), resample=LANCZOS)
        ref_image = pil2tensor(scaled_pil)

        assert (
            w == h and w == ref_image.shape[2] and h == ref_image.shape[1]
        ), f"Ref image size {ref_image.shape} must match {(w, h)} and must be square."

        result_image = self._tensor_paste(
            base_image.clone(), ref_image.to(base_image.device), (tl_x, tl_y)
        )

        return (result_image,)

    def _tensor_check_mask(self, mask):
        if mask.ndim != 4:
            raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
        if mask.shape[-1] != 1:
            raise ValueError(
                f"Expected 1 channel for mask, but found {mask.shape[-1]} channels"
            )
        return

    def _tensor_check_image(self, image):
        if image.ndim != 4:
            raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
        if image.shape[-1] not in (1, 3, 4):
            raise ValueError(
                f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels"
            )
        return

    def _tensor_mask(self, mask):
        """Return NHWC torch.Tenser from ndim == 2 or 4 `np.ndarray` or `torch.Tensor`"""
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if mask.ndim == 2:
            mask = mask[None, ..., None]
        elif mask.ndim == 3:
            mask = mask[..., None]

        self._tensor_check_mask(mask)
        return mask

    def _resize_mask(self, mask, size):
        resized_mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0), size=size, mode="bilinear", align_corners=False
        )
        return resized_mask.squeeze(0)

    def _tensor_paste(self, image1, image2, left_top):
        self._tensor_check_image(image1)
        self._tensor_check_image(image2)

        x, y = left_top
        _, h1, w1, _ = image1.shape
        _, h2, w2, _ = image2.shape

        # calculate image patch size
        w = min(w1, x + w2) - x
        h = min(h1, y + h2) - y

        # If the patch is out of bound, nothing to do!
        if w <= 0 or h <= 0:
            return

        image1[:, y : y + h, x : x + w, :] = image2[:, :h, :w, :]
        return image1


class GFPGAN(GFPGANer):
    def __init__(
        self,
        model_path,
        upscale=2,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
        device=None,
    ):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

        # initialize the GFP-GAN
        if arch == "clean":
            from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )

        elif arch == "RestoreFormer":
            from gfpgan.archs.restoreformer_arch import RestoreFormer

            self.gfpgan = RestoreFormer()

        else:
            raise Exception(f"Unsupported GFPGAN arch: {arch}")

        # initialize face helper
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            use_parse=True,
            device=self.device,
            model_rootpath=os.path.join(folder_paths.models_dir, "face_detection"),
        )

        loadnet = torch.load(model_path)
        if "params_ema" in loadnet:
            keyname = "params_ema"
        else:
            keyname = "params"

        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)


class FaceRestoreModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (["GFPGANv1_4"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "doit"
    CATEGORY = "Moriverse/io"

    def doit(self, type):
        if type == "GFPGANv1_4":
            arch = "clean"
            channel_multiplier = 2
            model_name = "GFPGANv1.4"

        else:
            raise Exception(f"Type of GFPGAN model not supported: {type}")

        model_path = os.path.join(
            folder_paths.models_dir,
            "face_restore",
            f"{model_name}.pth",
        )
        model = GFPGAN(
            model_path=model_path, arch=arch, channel_multiplier=channel_multiplier
        )

        return (model,)


class RestoreFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "images": ("IMAGE",),
            },
            "optional": {
                "weight": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"
    CATEGORY = "Moriverse/ops"

    def doit(self, model: GFPGAN, images, weight=0.5):
        result = []
        for image in images:
            image = np.clip(255.0 * image.cpu().numpy(), 0, 255)
            image = image.astype(np.uint8)
            image = image[:, :, ::-1]  # RGB -> BGR

            _, _, restored = model.enhance(
                image, has_aligned=False, paste_back=True, weight=weight
            )

            if restored is not None:
                restored = np.array(restored).astype(np.float32) / 255.0
                restored = restored[:, :, ::-1].copy()  # BGR -> RGB
                restored = torch.from_numpy(restored)
                result.append(restored)

        result = torch.stack(result, dim=0)
        return (result,)


class MediaPipeSegmenter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"
    CATEGORY = "Moriverse/ops"

    def doit(self, images, threshold=0.5):
        model = MediapipeSegmenter(
            model_path=os.path.join(
                folder_paths.models_dir,
                "mediapipe",
                f"selfie_segmenter.tflite",
            )
        )

        if isinstance(images, torch.Tensor):
            images = tensor2np(images)

        masks = []
        for image in images:
            mask = model.detect(image, threshold=threshold)
            mask = np2tensor(mask).unsqueeze(0)

            masks.append(mask)

        masks = torch.stack(masks)
        return (masks,)