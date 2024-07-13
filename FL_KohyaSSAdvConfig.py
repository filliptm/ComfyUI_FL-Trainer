import importlib
from . import FL_train_core
import os

class FL_KohyaSSAdvConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "xformers": (["enable", "disable"], {"default": "enable"}),
                "sdpa": (["enable", "disable"], {"default": "disable"}),
                "fp8_base": (["enable", "disable"], {"default": "disable"}),
                "mixed_precision": (["no", "fp16", "bf16"], {"default": "fp16"}),
                "gradient_accumulation_steps": ("INT", {"default": 1}),
                "gradient_checkpointing": (["enable", "disable"], {"default": "disable"}),
                "cache_latents": (["enable", "disable"], {"default": "enable"}),
                "cache_latents_to_disk": (["enable", "disable"], {"default": "enable"}),
                "network_dim": ("INT", {"default": 16}),
                "network_alpha": ("INT", {"default": 8}),
                "network_module": ([
                    "networks.lora",
                    "networks.dylora",
                    "networks.oft",
                ], {"default": "networks.lora"}),
                "network_train_unet_only": (["enable", "disable"], {"default": "enable"}),
                "lr_scheduler": ([
                    "linear", "cosine", "cosine_with_restarts", "polynomial",
                    "constant", "constant_with_warmup", "adafactor"
                ], {"default": "cosine"}),
                "lr_scheduler_num_cycles": ("INT", {"default": 1}),
                "optimizer_type": ([
                    "AdamW", "AdamW8bit", "PagedAdamW", "PagedAdamW8bit",
                    "PagedAdamW32bit", "Lion8bit", "PagedLion8bit", "Lion",
                    "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "DAdaptAdaGrad",
                    "DAdaptAdam", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion",
                    "DAdaptSGD", "AdaFactor"
                ], {"default": "AdamW"}),
                "lr_warmup_steps": ("INT", {"default": 0}),
                "unet_lr": ("STRING", {"default": ""}),
                "text_encoder_lr": ("STRING", {"default": ""}),
                "shuffle_caption": (["enable", "disable"], {"default": "disable"}),
                "save_precision": (["float", "fp16", "bf16"], {"default": "fp16"}),
                "persistent_data_loader_workers": (["enable", "disable"], {"default": "enable"}),
                "no_metadata": (["enable", "disable"], {"default": "enable"}),
                "noise_offset": ("FLOAT", {"default": 0.1}),
                "no_half_vae": (["enable", "disable"], {"default": "enable"}),
                "lowram": (["enable", "disable"], {"default": "disable"}),
            },
        }

    RETURN_TYPES = ("FL_TT_SS_AdvConfig",)
    RETURN_NAMES = ("advanced_config",)

    FUNCTION = "start"
    CATEGORY = "🏵️Fill Nodes/Training"

    def start(self, **kwargs):
        importlib.reload(FL_train_core)
        return FL_train_core.FL_KohyaSSAdvConfig_call(kwargs)