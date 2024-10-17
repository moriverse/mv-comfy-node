from .node_impl import *

NODE_CLASS_MAPPINGS = {
    "[MV]LoadSingleImageURL": LoadImageUrl,
    "[MV]LoadImagesFromURL": LoadImagesFromUrl,
    "[MV]FaceDetector": FaceDetectorForEach,
    "[MV]CropFace": CropFace,
    "[MV]PasteFace": PasteFace,
    "[MV]RestoreFace": RestoreFace,
    "[MV]FaceRestoreLoader": FaceRestoreModelLoader,
    "[MV]MediaPipeSegmenter": MediaPipeSegmenter,
    "[MV]Blip": Blip,
    "[MV]ImageContrast": ImageContrast,
    "[MV]ImageBrightness": ImageBrightness,
    "[MV]InsightFaceLoader": InsightFaceLoader,
    "[MV]GetCroppedFace": GetCroppedFace,
    "[MV]Text": Text,
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
    "[MV]Blip": "Blip",
    "[MV]ImageContrast": "Image Contrast",
    "[MV]ImageBrightness": "Image Brightness",
    "[MV]InsightFaceLoader": "InsightFace Loader",
    "[MV]GetCroppedFace": "Get Cropped Face",
    "[MV]Text": "Text",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
