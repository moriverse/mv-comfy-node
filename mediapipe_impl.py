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
        images,
        threshold=0.5,
        confidence_mask_index=0,  # Default person index at 0.
    ):
        masks = []
        try:
            options = ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.IMAGE,
            )

            with ImageSegmenter.create_from_options(options) as segmenter:
                for image in images:
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=image,
                    )
                    segmentation_result = segmenter.segment(mp_image)
                    print(segmentation_result)
                    confidence_mask = segmentation_result.confidence_masks[
                        confidence_mask_index
                    ]

                    binary_mask = confidence_mask.numpy_view() > threshold
                    masks.append((binary_mask * 255).astype(np.uint8))

        except Exception as e:
            print(f"MediapipeDetector: unable to execute.\n{e}")

        return masks
