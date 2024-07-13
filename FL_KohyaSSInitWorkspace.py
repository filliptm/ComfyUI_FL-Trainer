import importlib
from .FL_train_utils import Utils
from . import FL_train_core
import os

class FL_KohyaSSInitWorkspace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_name": ("STRING", {"default": ""}),
                "branch": ("STRING", {"default": "71e2c91330a9d866ec05cdd10584bbb962896a99"}),
                "source": ([
                    "github",
                    "githubfast",
                    "521github",
                    "kkgithub",
                ], {"default": "github"}),
                "seed": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("FL_TT_SS_WorkspaceConfig",)
    RETURN_NAMES = ("workspace_config",)

    FUNCTION = "start"
    CATEGORY = "🏵️Fill Nodes/Training"

    def start(self, **kwargs):
        importlib.reload(FL_train_core)
        return FL_train_core.FL_KohyaSSInitWorkspace_call(kwargs)