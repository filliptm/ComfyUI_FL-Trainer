import os
import importlib
import folder_paths
from .FL_train_utils import AlwaysEqualProxy, Utils


class FL_KohyaSSTrain:

    train_config_template_dir = os.path.join(
        os.path.dirname(__file__), "configs", "kohya_ss_lora"
    )

    @classmethod
    def INPUT_TYPES(s):
        loras = [
            "latest",
            "empty",
        ]

        workspaces_dir = os.path.join(
            folder_paths.output_directory, "FL_train_workspaces")

        workspaces_loras = []
        for root, dirs, files in os.walk(workspaces_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if root.endswith("output"):
                for file in files:
                    if file.endswith(".safetensors"):
                        workspaces_loras.append(
                            os.path.join(root, file)
                        )

        workspaces_loras = sorted(
            workspaces_loras, key=lambda x: os.path.getctime(x), reverse=True)

        comfyui_full_loras = []
        comfyui_loras = folder_paths.get_filename_list("loras")
        for lora in comfyui_loras:
            lora_path = folder_paths.get_full_path("loras", lora)
            comfyui_full_loras.append(lora_path)

        comfyui_full_loras = sorted(
            comfyui_full_loras, key=lambda x: os.path.getctime(x), reverse=True)

        loras = loras + workspaces_loras + comfyui_full_loras

        train_config_templates = Utils.listdir(s.train_config_template_dir)

        priority = [
            "lora",
            "1_2",
            "1_1"
        ]

        train_config_templates = [os.path.splitext(x)[0]
                                  for x in train_config_templates]

        def priority_sort(x):
            for p in priority:
                if x.find(p) != -1:
                    return priority.index(p)
            return 999

        train_config_templates = sorted(
            train_config_templates, key=priority_sort)

        return {
            "required": {
                "workspace_config": ("FL_TT_SS_WorkspaceConfig",),
                "train_config_template": (train_config_templates,),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                "max_train_steps": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "max_train_epochs": ("INT", {"default": 100, "min": 0, "max": 0x7fffffff}),
                "save_every_n_epochs": ("INT", {"default": 1}),
                "learning_rate": ("STRING", {"default": "0.0001"}),
                "base_lora": (loras, {"default": "latest"}),
                "sample_prompt": ("STRING", {"default": "", "dynamicPrompts": False, "multiline": False}),
                "advanced_config": ("FL_TT_SS_AdvConfig",),
            },
            "optional": {
                "caption_completed_flag": (AlwaysEqualProxy("*"),),
            },
        }

    sample_generate = "enable"

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "start"
    CATEGORY = "🏵️Fill Nodes/Training"

    def start(self, **kwargs):
        from . import FL_train_core
        importlib.reload(FL_train_core)
        return FL_train_core.FL_KohyaSSTrain_call(kwargs)