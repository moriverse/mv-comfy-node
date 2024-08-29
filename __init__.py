from .node_impl import *
from .clip_tools import *

NODE_CLASS_MAPPINGS = {
    "[MV]LoadSingleImageURL": LoadImageUrl,
    "[MV]LoadImagesFromURL": LoadImagesFromUrl,
    "[MV]FaceDetector": FaceDetectorForEach,
    "[MV]CropFace": CropFace,
    "[MV]PasteFace": PasteFace,
    "[MV]RestoreFace": RestoreFace,
    "[MV]FaceRestoreLoader": FaceRestoreModelLoader,
    "[MV]MediaPipeSegmenter": MediaPipeSegmenter,
    "[MV]ClipInterrogator": CLIP_Interrogator,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "[MV]LoadSingleImageURL": "Load Single Image URL",
    "[MV]LoadImagesFromURL": "Load Images From URL",
    "[MV]FaceDetector": "Face Detector",
    "[MV]CropFace": "Crop Face",
    "[MV]PasteFace": "Paste Face",
    "[MV]RestoreFace": "Restore Face",
    "[MV]FaceRestoreLoader": "Face Restore Loader",
    "[MV]MediaPipeSegmenter": "Mediapipe Segmenter",
    "[MV]ClipInterrogator": "Clip Interrogator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
