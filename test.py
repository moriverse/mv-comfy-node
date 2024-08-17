import numpy as np

from PIL import Image

# model_path = '/home/doge/comfy/ComfyUI/models/mediapipe/deeplab_v3.tflite'
# model_path = '/home/doge/comfy/ComfyUI/models/mediapipe/hair_segmenter.tflite'
# model_path = (
#     "/home/doge/comfy/ComfyUI/models/mediapipe/selfie_multiclass_256x256.tflite"
# )
model_path = "/home/doge/comfy/ComfyUI/models/mediapipe/selfie_segmenter.tflite"

image_file_name = "/home/doge/Downloads/20240724-213302.jpg"

from mediapipe_impl import MediapipeSegmenter

model = MediapipeSegmenter(model_path=model_path)

numpy_image = np.array(Image.open(image_file_name))
masks = model.detect([numpy_image], threshold=0.5)

Image.fromarray(masks[0], mode="L").save("debug.png")
