# __init__.py

from .FL_KohyaSSInitWorkspace import FL_KohyaSSInitWorkspace
from .FL_KohyaSSDatasetConfig import FL_KohyaSSDatasetConfig
from .FL_KohyaSSAdvConfig import FL_KohyaSSAdvConfig
from .FL_KohyaSSTrain import FL_KohyaSSTrain
from .FL_LoadImagesFromDirectoryPath import FL_LoadImagesFromDirectoryPath
from .FL_Kohya_EasyTrain import FL_Kohya_EasyTrain
#==============================================================================
#==============================================================================
# from .FL_SliderLoraInitWorkspace import FL_SliderLoraInitWorkspace
# from .FL_SliderLoraDatasetConfig import FL_SliderLoraDatasetConfig
# from .FL_SliderLoraAdvConfig import FL_SliderLoraAdvConfig
# from .FL_SliderLoraTrain import FL_SliderLoraTrain


NODE_CLASS_MAPPINGS = {
    "FL_KohyaSSInitWorkspace": FL_KohyaSSInitWorkspace,
    "FL_KohyaSSDatasetConfig": FL_KohyaSSDatasetConfig,
    "FL_KohyaSSAdvConfig": FL_KohyaSSAdvConfig,
    "FL_KohyaSSTrain": FL_KohyaSSTrain,
    "FL_LoadImagesFromDirectoryPath": FL_LoadImagesFromDirectoryPath,
    "FL_Kohya_EasyTrain": FL_Kohya_EasyTrain,
#==============================================================================
#==============================================================================
    # "FL_SliderLoraInitWorkspace": FL_SliderLoraInitWorkspace,
    # "FL_SliderLoraDatasetConfig": FL_SliderLoraDatasetConfig,
    # "FL_SliderLoraAdvConfig": FL_SliderLoraAdvConfig,
    # "FL_SliderLoraTrain": FL_SliderLoraTrain
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_KohyaSSInitWorkspace": "FL Kohya Workspace",
    "FL_KohyaSSDatasetConfig": "FL Kohya Dataset Config",
    "FL_KohyaSSAdvConfig": "FL Kohya Adv Config",
    "FL_KohyaSSTrain": "FL Kohya Train",
    "FL_LoadImagesFromDirectoryPath": "FL Kohya Data Loader",
    "FL_Kohya_EasyTrain": "FL Kohya Easy Train",
#==============================================================================
#==============================================================================
    # "FL_SliderLoraInitWorkspace": "FL Slider LoRA Init Workspace",
    # "FL_SliderLoraDatasetConfig": "FL Slider LoRA Dataset Config",
    # "FL_SliderLoraAdvConfig": "FL Slider LoRA Advanced Config",
    # "FL_SliderLoraTrain": "FL Slider LoRA Train"
}
ascii_art = """

MACHINE DELUSIONS
TRAINER LOADED
                                                                                        
"""
print(ascii_art)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]