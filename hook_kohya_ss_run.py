import os

os.system("title hook_kohya_ss_run")
import random
import time

import torch
import logging
import sys
import json
import importlib
import argparse
import toml


current_dir = os.path.dirname(os.path.abspath(__file__))
kohya_ss_dir = os.path.join(current_dir, "kohya_ss_lora")
if kohya_ss_dir not in sys.path:
    sys.path.append(kohya_ss_dir)


def config2args(train_parser: argparse.ArgumentParser, config):

    config_args_list = []
    for key, value in config.items():
        if type(value) == bool:
            if value:
                config_args_list.append(f"--{key}")
        else:
            config_args_list.append(f"--{key}")
            config_args_list.append(str(value))
    args = train_parser.parse_args(config_args_list)
    return args


from PIL import Image


import numpy as np
import tempfile
import safetensors.torch


import sys
sys.path.append(os.path.dirname(__file__))
try:
    import hook_kohya_ss_utils
except:
    from . import hook_kohya_ss_utils

other_config = {}
original_save_model = None


train_config = {}

sample_images_pipe_class = None


def utils_sample_images(*args, **kwargs):
    return sample_images(None, *args, **kwargs)


def get_datasets():
    import library.config_util
    user_config = library.config_util.load_user_config(
        train_config.get("dataset_config", None))
    datasets = user_config.get("datasets", [])
    if len(datasets) == 0:
        return None
    return datasets[0]


def sample_images(self, *args, **kwargs):
    #  accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet
    accelerator = args[0]
    cmd_args = args[1]
    epoch = args[2]
    global_step = args[3]
    device = args[4]
    vae = args[5]
    tokenizer = args[6]
    text_encoder = args[7]
    unet = args[8]

    # print(f"sample_images: args = {args}")
    # print(f"sample_images: kwargs = {kwargs}")

    controlnet = kwargs.get("controlnet", None)

    if epoch is not None and cmd_args.save_every_n_epochs is not None and epoch % cmd_args.save_every_n_epochs == 0:

        datasets = get_datasets()
        resolution = datasets.get("resolution", (512, 512))
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        height, width = resolution
        print(f"sample_images: height = {height}, width = {width}")

        prompt_dict_list = other_config.get("prompt_dict_list", [])
        if len(prompt_dict_list) == 0:
            sample_prompt = other_config.get("sample_prompt", None)
            if sample_prompt is not None:
                seed = other_config.get("seed", 0)
                prompt_dict = {
                    "controlnet_image": other_config.get("controlnet_image", None),
                    "prompt": other_config.get("sample_prompt", ""),
                    "seed": seed,
                    "negative_prompt": "",
                    "enum": 0,
                    "sample_sampler": "euler_a",
                    "sample_steps": 20,
                    "scale": 5.0,
                    "height": height,
                    "width": width,
                }
                #
                prompt_dict_list.append(prompt_dict)
        else:
            for i, prompt_dict in enumerate(prompt_dict_list):
                if prompt_dict.get("controlnet_image", None) is None:
                    prompt_dict["controlnet_image"] = None
                if prompt_dict.get("seed", None) is None:
                    prompt_dict["seed"] = 0
                if prompt_dict.get("negative_prompt", None) is None:
                    prompt_dict["negative_prompt"] = ""
                if prompt_dict.get("enum", None) is None:
                    prompt_dict["enum"] = i

        if prompt_dict_list is not None and len(prompt_dict_list) > 0:
            hook_kohya_ss_utils.generate_image(
                pipe_class=sample_images_pipe_class,
                cmd_args=cmd_args,
                accelerator=accelerator,
                epoch=epoch,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                vae=vae,
                prompt_dict_list=prompt_dict_list,
                controlnet=controlnet,
            )

    LOG({
        "type": "sample_images",
        "global_step": global_step,
        "total_steps": cmd_args.max_train_steps,
        # "latent": noise_pred_latent_path,
    })


def run_lora_sd1_5():
    hook_kohya_ss_utils.hook_kohya_ss()

    import train_network
    train_network.NetworkTrainer.sample_images = sample_images

    import library.train_util
    global sample_images_pipe_class
    sample_images_pipe_class = library.train_util.StableDiffusionLongPromptWeightingPipeline

    trainer = train_network.NetworkTrainer()
    train_args = config2args(train_network.setup_parser(), train_config)

    LOG({
        "type": "start_train",
    })
    trainer.train(train_args)


def run_lora_sdxl():
    hook_kohya_ss_utils.hook_kohya_ss()

    import sdxl_train_network
    sdxl_train_network.SdxlNetworkTrainer.sample_images = sample_images

    import library.sdxl_train_util
    global sample_images_pipe_class
    sample_images_pipe_class = library.sdxl_train_util.SdxlStableDiffusionLongPromptWeightingPipeline

    trainer = sdxl_train_network.SdxlNetworkTrainer()
    train_args = config2args(sdxl_train_network.setup_parser(), train_config)

    LOG({
        "type": "start_train",
    })
    trainer.train(train_args)


from types import SimpleNamespace


class SimpleNamespaceCNWarrper(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(kwargs)  # or self.__dict__ = kwargs
        self.__dict__["mid_block_type"] = "UNetMidBlock2DCrossAttn"
        self.__dict__["_diffusers_version"] = "0.6.0"
        self.__iter__ = lambda: iter(kwargs.keys())
    # is not iterable

    def __iter__(self):
        return iter(self.__dict__.keys())
    # object has no attribute 'num_attention_heads'

    def __getattr__(self, name):
        return self.__dict__.get(name, None)


def run_controlnet_sd1_5():
    import types
    types.SimpleNamespace = SimpleNamespaceCNWarrper
    hook_kohya_ss_utils.hook_kohya_ss()

    import train_controlnet

    import library.train_util
    library.train_util.sample_images = utils_sample_images

    global sample_images_pipe_class
    sample_images_pipe_class = library.train_util.StableDiffusionLongPromptWeightingPipeline

    train_args = config2args(train_controlnet.setup_parser(), train_config)

    LOG({
        "type": "start_train",
    })

    train_controlnet.train(train_args)


def run_lora_hunyuan1_2():
    hook_kohya_ss_utils.hook_kohya_ss()

    import hunyuan_train_network

    hunyuan_train_network.HunYuanNetworkTrainer.sample_images = sample_images


    import hook_kohya_ss_hunyuan_pipe
    global sample_images_pipe_class
    sample_images_pipe_class = hook_kohya_ss_hunyuan_pipe.HuanYuanDiffusionLongPromptWeightingPipeline
    import library.hunyuan_utils

    print(json.dumps(other_config, indent=4))
    hunyuan_models_config = other_config.get("hunyuan_models_config", None)

    from transformers import (
        AutoTokenizer,
        T5Tokenizer,
        BertModel,
        BertTokenizer,
    )
    from diffusers import AutoencoderKL, LMSDiscreteScheduler

    def hunyuan_load_tokenizers():
        tokenizer = AutoTokenizer.from_pretrained(
            hunyuan_models_config["tokenizer_path"],
            local_files_only=True,
        )
        tokenizer.eos_token_id = tokenizer.sep_token_id
        t5_encoder_path = hunyuan_models_config.get("t5_encoder_path", None)
        if t5_encoder_path == "none":
            t5_encoder_path = None
        tokenizer2 = None
        if t5_encoder_path is not None:
            tokenizer2 = T5Tokenizer.from_pretrained(
                t5_encoder_path,
                local_files_only=True,
            )

        return [tokenizer, tokenizer2]

    library.hunyuan_utils.load_tokenizers = hunyuan_load_tokenizers

    def hunyuan_load_model(model_path: str, dtype=torch.float16, device="cuda", use_extra_cond=False, dit_path=None):

        dit_path = hunyuan_models_config.get("unet_path", None)

        import library.hunyuan_models
        MT5Embedder = library.hunyuan_models.MT5Embedder
        HunYuanDiT = library.hunyuan_models.HunYuanDiT
        BertModel = library.hunyuan_models.BertModel
        DiT_g_2 = library.hunyuan_models.DiT_g_2

        denoiser, patch_size, head_dim = DiT_g_2(
            input_size=(128, 128), use_extra_cond=use_extra_cond)
        if dit_path is not None:
            state_dict = torch.load(dit_path)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            state_dict = torch.load(os.path.join(
                model_path, "denoiser/pytorch_model_module.pt"))
        denoiser.load_state_dict(state_dict)
        denoiser.to(device).to(dtype)

        clip_tokenizer = AutoTokenizer.from_pretrained(
            hunyuan_models_config["tokenizer_path"],
            local_files_only=True,
        )
        clip_tokenizer.eos_token_id = 2
        clip_encoder = (
            BertModel.from_pretrained(
                hunyuan_models_config["text_encoder_path"],
                local_files_only=True,
            ).to(device).to(dtype)
        )

        t5_encoder_path = hunyuan_models_config.get("t5_encoder_path", None)
        if t5_encoder_path == "none":
            t5_encoder_path = None
        mt5_embedder = None
        if t5_encoder_path is not None:
            mt5_embedder = (
                MT5Embedder(
                    model_dir=hunyuan_models_config["t5_encoder_path"],
                    torch_dtype=dtype,
                    max_length=256)
                .to(device)
                .to(dtype)
            )
        else:

            batch_size = train_args.train_batch_size
            import library.config_util
            user_config = library.config_util.load_user_config(
                train_args.dataset_config)
            datasets = user_config.get("datasets", [])
            if len(datasets) > 0:
                batch_size = datasets[0].get("batch_size", batch_size)
            mt5_embedder = (
                hook_kohya_ss_utils.CustomizeMT5Embedder(
                    batch_size=batch_size,
                )
                .to(device)
                .to(dtype)
            )

        vae = (
            AutoencoderKL.from_pretrained(
                hunyuan_models_config["vae_ema_path"],
                local_files_only=True,
            )
            .to(device)
            .to(dtype)
        )
        vae.requires_grad_(False)
        return (
            denoiser,
            patch_size,
            head_dim,
            clip_tokenizer,
            clip_encoder,
            mt5_embedder,
            vae,
        )

    library.hunyuan_utils.load_model = hunyuan_load_model

    trainer = hunyuan_train_network.HunYuanNetworkTrainer()
    train_args = config2args(
        hunyuan_train_network.setup_parser(), train_config)
    print(f"train_args = {train_args}")

    LOG({
        "type": "start_train",
    })
    trainer.train(train_args)


func_map = {
    "run_lora_sd1_5": run_lora_sd1_5,
    "run_lora_sdxl": run_lora_sdxl,
    "run_controlnet_sd1_5": run_controlnet_sd1_5,
    "run_lora_hunyuan1_2": run_lora_hunyuan1_2,
}


import requests


def LOG(log):
    try:
        resp = requests.request("post", f"http://127.0.0.1:{master_port}/log", data=json.dumps(log), headers={
                                "Content-Type": "application/json"})
        if resp.status_code != 200:
            # raise Exception(f"LOG failed: {resp.text}")
            print(f"LOG failed: {resp.text}")
    except Exception as e:
        print(f"LOG failed: {e}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--sys_path", type=str, default="")
        parser.add_argument("--config", type=str, default="")
        parser.add_argument("--train_func", type=str, default="")
        parser.add_argument("--master_port", type=int, default=0)
        args = parser.parse_args()

        master_port = args.master_port

        print(f"master_port = {master_port}")

        sys_path = args.sys_path
        if sys_path == "":
            sys_path = kohya_ss_dir
        sys.path.append(sys_path)

        config_file = args.config
        if config_file == "":
            raise Exception("train_config is empty")

        global_config = {}
        with open(config_file, "r") as f:
            _global_config = f.read()
            global_config = json.loads(_global_config)

        train_config = global_config.get("train_config")
        print(f"""=======================train_config=======================
    {json.dumps(train_config, indent=4, ensure_ascii=False)}
            """)

        other_config = global_config.get("other_config", {})
        print(f"""=======================other_config=======================
    {json.dumps(other_config, indent=4, ensure_ascii=False)}
            """)

        train_func = args.train_func
        if train_func == "":
            raise Exception("train_func is empty")

        print(f"train_func = {train_func}")

        time.sleep(2)
        LOG({
            "type": "Read configuration completed!",
        })

        func_map[train_func]()
    except Exception as e:
        print(f"Exception: {e}")
        if sys.platform == "win32":
            input("Press Enter to continue...")