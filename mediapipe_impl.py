import mediapipe as mp
import numpy as np

from collections import namedtuple

SEG = namedtuple(
    "SEG",
    [
        "cropped_image",
        "cropped_mask",
        "confidence",
        "crop_region",
        "bbox",
        "label",
        "control_net_wrapper",
    ],
    defaults=[None],
)

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class MediapipeSegmenter:

    def __init__(self, model_path):
        self.model_path = model_path

    def detect(
        self,
        image,
        threshold=0.5,
        confidence_mask_indexes=[0],  # Default person index at 0.
    ):
        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
        )

        with ImageSegmenter.create_from_options(options) as segmenter:
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image.copy(),
            )
            segmentation_result = segmenter.segment(mp_image)

            h, w, _ = image.shape
            combined_mask = np.zeros((h, w), dtype=np.bool_)
            for index in confidence_mask_indexes:
                confidence_mask = segmentation_result.confidence_masks[index]
                binary_mask = confidence_mask.numpy_view() > threshold

                combined_mask = np.bitwise_or(combined_mask, binary_mask)

            return (combined_mask * 255).astype(np.uint8)