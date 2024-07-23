import importlib
from . import FL_train_core
import os
import folder_paths
from .FL_train_utils import Utils
import subprocess
import sys

class FL_Kohya_EasyTrain:
    train_config_template_dir = os.path.join(
        os.path.dirname(__file__), "configs", "kohya_ss_lora"
    )

    @classmethod
    def INPUT_TYPES(s):
        train_config_templates = Utils.listdir(s.train_config_template_dir)
        train_config_templates = [os.path.splitext(x)[0] for x in train_config_templates]

        return {
            "required": {
                "lora_name": ("STRING", {"default": "my_lora"}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048}),
                "train_config_template": (train_config_templates, {"default": "lora_sd1_5"}),
                "num_repeats": ("INT", {"default": 30, "min": 1}),
                "image_directory": ("STRING", {"default": "path/to/images+captions.txt"}),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "sample_prompt": ("STRING", {"default": "Sampling prompt here", "multiline": True}),
                "xformers": (["enable", "disable"], {"default": "disable"}),
                "lowvram": (["enable", "disable"], {"default": "disable"}),
                "learning_rate": ("FLOAT", {"default": 0.0001, "min": 0.0000001, "max": 0.1, "step": 0.0000001}),
                "epochs": ("INT", {"default": 10, "min": 1, "max": 1000}),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "start"
    CATEGORY = "🏵️Fill Nodes/Training"

    def start(self, lora_name, resolution, train_config_template, num_repeats, image_directory, ckpt_name,
              sample_prompt, xformers, lowvram, learning_rate, epochs):
        importlib.reload(FL_train_core)

        if not lora_name.strip():
            raise ValueError("LoRA name is required. Please provide a name for your LoRA.")

        # Clone or update the repository
        self.clone_or_update_repo()

        # Initialize workspace
        workspaces_dir = os.path.join(folder_paths.output_directory, "FL_train_workspaces")
        os.makedirs(workspaces_dir, exist_ok=True)
        workspace_dir = os.path.join(workspaces_dir, lora_name)
        os.makedirs(workspace_dir, exist_ok=True)

        workspace_config = {
            "workspace_name": lora_name,
            "workspace_dir": workspace_dir,
        }

        # Load images and set up dataset configuration
        images, captions = self.load_images(image_directory)
        dataset_config = FL_train_core.FL_ImageSelecter_call({
            "workspace_config": workspace_config,
            "images": images,
            "captions": captions,
            "enable_bucket": "enable",
            "resolution": resolution,
            "num_repeats": num_repeats,
            "caption_extension": ".txt",
            "batch_size": 1,
            "force_clear": "disable",
            "force_clear_only_images": "disable",
            "image_format": "jpg",
            "dataset_config_extension": ".json"
        })

        # Configure advanced settings
        adv_config = FL_train_core.FL_KohyaSSAdvConfig_call({
            "xformers": xformers,
            "sdpa": "disable",
            "fp8_base": "disable",
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": "disable",
            "cache_latents": "enable",
            "cache_latents_to_disk": "enable",
            "network_dim": 64,
            "network_alpha": 32,
            "network_module": "networks.lora",
            "network_train_unet_only": "disable",
            "lr_scheduler": "constant",
            "lr_scheduler_num_cycles": 1,
            "optimizer_type": "AdaFactor",
            "lr_warmup_steps": 0,
            "unet_lr": str(learning_rate),
            "text_encoder_lr": str(learning_rate / 2),
            "shuffle_caption": "enable",
            "save_precision": "fp16",
            "persistent_data_loader_workers": "disable",
            "no_metadata": "disable",
            "noise_offset": 0.05,
            "no_half_vae": "enable",
            "lowram": lowvram
        })

        # Start training
        train_args = {
            "workspace_config": workspace_config,
            "train_config_template": train_config_template,
            "ckpt_name": ckpt_name,
            "max_train_steps": 1000000,
            "max_train_epochs": epochs,
            "save_every_n_epochs": 1,
            "learning_rate": str(learning_rate),
            "base_lora": "empty",
            "sample_prompt": sample_prompt,
            "advanced_config": adv_config[0],
        }

        return FL_train_core.FL_KohyaSSTrain_call(train_args)

    def load_images(self, directory):
        images = []
        captions = []
        if os.path.exists(directory):
            files = Utils.listdir(directory)
            image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".webp", ".jpeg"))]
            for image_file in image_files:
                image_path = os.path.join(directory, image_file)
                caption_path = os.path.splitext(image_path)[0] + ".txt"
                if os.path.exists(caption_path):
                    with open(caption_path, 'r', encoding='utf-8') as f:
                        captions.append(f.read().strip())
                    images.append(Utils.pil2tensor(Utils.loadImage(image_path)))
        return Utils.list_tensor2tensor(images), captions

    def clone_or_update_repo(self):
        FL_dir = Utils.get_minus_zone_models_path()
        kohya_ss_lora_dir = os.path.join(FL_dir, "train_tools", "kohya_ss_lora")
        repo_url = "https://github.com/kohya-ss/sd-scripts"
        branch = "main"  # You can change this to a specific branch if needed

        if not os.path.exists(kohya_ss_lora_dir):
            # Clone the repository if it doesn't exist
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, kohya_ss_lora_dir],
                check=True
            )
        else:
            # Update the existing repository
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", branch],
                cwd=kohya_ss_lora_dir,
                check=True
            )
            subprocess.run(
                ["git", "checkout", branch],
                cwd=kohya_ss_lora_dir,
                check=True
            )
            subprocess.run(
                ["git", "pull", "origin", branch],
                cwd=kohya_ss_lora_dir,
                check=True
            )

        # Ensure the repo directory is in sys.path
        if kohya_ss_lora_dir not in sys.path:
            sys.path.append(kohya_ss_lora_dir)