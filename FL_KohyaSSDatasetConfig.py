import importlib
from .FL_train_utils import Utils
from . import FL_train_core
import os

class FL_KohyaSSDatasetConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "workspace_config": ("FL_TT_SS_WorkspaceConfig",),
                "images": ("IMAGE",),
                "captions": ("STRING", {"forceInput": True}),
                "enable_bucket": (["enable", "disable"], {"default": "enable"}),
                "resolution": ("INT", {"default": 1024}),
                "num_repeats": ("INT", {"default": 1, "min": 1}),
                "caption_extension": ([".caption", ".txt"], {"default": ".caption"}),
                "batch_size": ("INT", {"default": 1, "min":1}),
                "force_clear": (["enable", "disable"], {"default": "disable"}),
                "force_clear_only_images": (["enable", "disable"], {"default": "disable"}),
                "image_format": (["png", "jpg", "webp"], {"default": "jpg"}),
                "dataset_config_extension": ([".toml", ".json"], {"default": ".json"}),
            },
            "optional": {
                "conditioning_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("workspace_images_dir",)

    FUNCTION = "start"
    CATEGORY = "🏵️Fill Nodes/Training"

    def start(self, **kwargs):
        importlib.reload(FL_train_core)
        return FL_train_core.FL_ImageSelecter_call(kwargs)