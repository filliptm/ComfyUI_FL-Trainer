from .FL_train_utils import Utils
import os
from PIL import Image

class FL_LoadImagesFromDirectoryPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "X://path/to/images"}),
                "caption_extension": ([".caption", ".txt"], {"default": ".caption"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "captions")

    FUNCTION = "start"

    CATEGORY = "🏵️Fill Nodes/Training"

    def start(self, directory, caption_extension):

        images = []
        captions = []
        if not os.path.exists(directory):
            return (Utils.list_tensor2tensor([]), [])

        files = Utils.listdir(directory)
        image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".webp", ".jpeg"))]

        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            caption_path = os.path.splitext(image_path)[0] + caption_extension

            if os.path.exists(caption_path):
                with open(caption_path, 'r', encoding='utf-8') as f:
                    captions.append(f.read().strip())
                pil_image = Image.open(image_path)
                images.append(Utils.pil2tensor(pil_image))

        return (Utils.list_tensor2tensor(images), captions)