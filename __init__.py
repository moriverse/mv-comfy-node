from .node_impl import *

NODE_CLASS_MAPPINGS = {
    "[MV]LoadImagesFromURL": LoadImagesFromUrl,
    "[MV]FaceDetector": FaceDetectorForEach,
    "[MV]CropFace": CropFace,
    "[MV]PasteFace": PasteFace,
    "[MV]RestoreFace": RestoreFace,
    "[MV]FaceRestoreLoader": FaceRestoreModelLoader,
    "[MV]MediaPipeSegmenter": MediaPipeSegmenter,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "[MV]LoadImagesFromURL": "Load Images From URL",
    "[MV]FaceDetector": "Face Detector",
    "[MV]CropFace": "Crop Face",
    "[MV]PasteFace": "Paste Face",
    "[MV]RestoreFace": "Restore Face",
    "[MV]FaceRestoreLoader": "Face Restore Loader",
    "[MV]MediaPipeSegmenter": "Mediapipe Segmenter",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
