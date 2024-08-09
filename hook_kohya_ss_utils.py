import argparse
import json
import os
from typing import *
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from transformers import CLIPTokenizer
import requests

_requests_get = requests.get

source_replacement_table = {
    "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml": os.path.join(
        os.path.dirname(__file__), "configs", "models_config", "stable-diffusion-v1.5", "v1-inference.yaml"),
    "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml": os.path.join(
        os.path.dirname(__file__), "configs", "models_config", "stable-diffusion-xl", "sd_xl_base.yaml"),
    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer_config.json": os.path.join(
        os.path.dirname(__file__), "configs", "models_config", "clip-vit-large-patch14", "tokenizer_config.json"),
}

source_replacement_dir = {
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main": os.path.join(
        os.path.dirname(__file__), "configs", "models_config", "stable-diffusion-xl-base-1.0"),
}

class DictWrapper:
    def __init__(self, d):
        self.d = d

    def __getattribute__(self, name: str):
        if name == "content":
            return self.d["content"]
        if name == "raise_for_status":
            return lambda: None
        if name == "json":
            return lambda: json.loads(self.d["content"])
        if name == "status_code":
            return self.d["status_code"]
        if name == "headers":
            return {
                "Location": self.d["Location"],
                "Content-Length": len(self.d["content"]),
            }
        if name == "request":
            return None
        return super().__getattribute__(name)

def request_wrapper(*args, **kwargs):
    url = args[1]
    print(f"request_wrapper requesting {url}")
    if url in source_replacement_table:
        with open(source_replacement_table[url], "rb") as f:
            return DictWrapper({
                "Location": url,
                "content": f.read(),
                "status_code": 200,
            })

    print(f"request_wrapper requesting {url} from original requests")
    return _requests_get(*args, **kwargs)

from requests import api
from requests import Session
last_request = api.request
original_session_request = Session.request
api.request = request_wrapper

def Session_request_wrapper(cls, method, url, **kwargs):
    if url.startswith("http://127.0.0.1"):
        return original_session_request(cls, method, url, **kwargs)
    if url in source_replacement_table:
        with open(source_replacement_table[url], "rb") as f:
            return DictWrapper({
                "Location": url,
                "content": f.read(),
                "status_code": 200,
            })

    for k, v in source_replacement_dir.items():
        if url.startswith(k):
            file_path = source_replacement_dir[k] + url[len(k):]
            with open(file_path, "rb") as f:
                return DictWrapper({
                    "Location": url,
                    "content": f.read(),
                    "status_code": 200,
                })
    raise NotImplementedError("Session.request is not supported")

Session.request = Session_request_wrapper

import diffusers.loaders.single_file
original_snapshot_download = diffusers.loaders.single_file.snapshot_download

def _snapshot_download(repo_id, *args, **kwargs):
    print(f"_snapshot_download: {repo_id}")
    if repo_id == "runwayml/stable-diffusion-v1-5":
        return os.path.join(
            os.path.dirname(__file__), "configs", "models_config", "stable-diffusion-v1-5",)
    if repo_id == "stabilityai/stable-diffusion-xl-base-1.0":
        return os.path.join(
            os.path.dirname(__file__), "configs", "models_config", "stable-diffusion-xl-base-1.0",)
    raise NotImplementedError("_snapshot_download is not supported")

diffusers.loaders.single_file.snapshot_download = _snapshot_download

original_load_target_model = None

def setup_logging(*args, **kwargs):
    pass

clip_large_tokenizer = None
clip_big_tokenizer = None

class TokenizersWrapper:
    typed = None
    model_max_length = 77

    def __init__(self, t):
        self.model_max_length = 77
        self.typed = t

    def __getattribute__(self, name: str):
        if name == "model_max_length":
            return 77
        try:
            typed = object.__getattribute__(self, "typed")
            if typed == "clip_large" and clip_large_tokenizer is not None:
                return clip_large_tokenizer.__getattribute__(name)
            if typed == "clip_big" and clip_big_tokenizer is not None:
                return clip_big_tokenizer.__getattribute__(name)
        except:
            pass

        return object.__getattribute__(self, name)

    def __call__(self, *args, **kargs):
        if self.typed == "clip_large":
            return clip_large_tokenizer(*args, **kargs)
        if self.typed == "clip_big":
            return clip_big_tokenizer(*args, **kargs)

        raise NotImplementedError(
            f"TokenizersWrapper: {self.typed} is not supported")

def load_tokenizers(*args, **kwargs):
    return TokenizersWrapper("clip_large")

def load_sdxl_tokenizers(*args, **kwargs):
    return [TokenizersWrapper("clip_large"), TokenizersWrapper("clip_big")]

original_conditional_loss = None

running_info = {}

def conditional_loss(*args, **kwargs):
    running_info["last_noise_pred"] = args[0]
    return original_conditional_loss(*args, **kwargs)

def decode_latents(vae, latents):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latents = latents.to(dtype=vae.dtype).to(device)
    vae = vae.to(device)
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()
    return image

def hook_kohya_ss():
    import library.utils
    import library.train_util
    import library.sdxl_train_util
    library.utils.setup_logging = setup_logging

    library.train_util.load_tokenizer = load_tokenizers
    library.sdxl_train_util.load_tokenizers = load_sdxl_tokenizers

    global original_load_target_model
    if original_load_target_model is None:
        original_load_target_model = library.train_util._load_target_model

    library.train_util._load_target_model = _load_target_model
    library.sdxl_train_util._load_target_model = _sdxl_load_target_model

    global original_conditional_loss
    if original_conditional_loss is None:
        original_conditional_loss = library.train_util.conditional_loss
    library.train_util.conditional_loss = conditional_loss

def _sdxl_load_target_model(
    name_or_path: str, vae_path: Optional[str], model_version: str, weight_dtype, device="cpu", model_dtype=None, *args, **kwargs
):
    import library.sdxl_model_util as sdxl_model_util
    import library.model_util as model_util
    import library.sdxl_original_unet as sdxl_original_unet
    import library.sdxl_train_util
    init_empty_weights = library.sdxl_train_util.init_empty_weights

    name_or_path = os.readlink(name_or_path) if os.path.islink(
        name_or_path) else name_or_path
    load_stable_diffusion_format = False

    if True:
        variant = "fp16" if weight_dtype == torch.float16 else None
        print(
            f"load Diffusers pretrained models: {name_or_path}, variant={variant}")
        try:
            try:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    name_or_path, local_files_only=True, safety_checker=None)
            except EnvironmentError as ex:
                raise ex
        except EnvironmentError as ex:
            print(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex

        text_encoder1 = pipe.text_encoder
        text_encoder2 = pipe.text_encoder_2

        if text_encoder1.dtype != torch.float32:
            text_encoder1 = text_encoder1.to(dtype=torch.float32)
        if text_encoder2.dtype != torch.float32:
            text_encoder2 = text_encoder2.to(dtype=torch.float32)

        vae = pipe.vae
        unet = pipe.unet
        global clip_large_tokenizer, clip_big_tokenizer
        clip_large_tokenizer = pipe.tokenizer
        clip_big_tokenizer = pipe.tokenizer_2
        del pipe

        state_dict = sdxl_model_util.convert_diffusers_unet_state_dict_to_sdxl(
            unet.state_dict())
        with init_empty_weights():
            unet = sdxl_original_unet.SdxlUNet2DConditionModel()
        sdxl_model_util._load_state_dict_on_device(
            unet, state_dict, device=device, dtype=model_dtype)
        print("U-Net converted to original U-Net")

        logit_scale = None
        ckpt_info = None

    if vae_path is not None:
        vae = model_util.load_vae(vae_path, weight_dtype)
        print("additional VAE loaded")

    return load_stable_diffusion_format, text_encoder1, text_encoder2, vae, unet, logit_scale, ckpt_info

def _load_target_model(args: argparse.Namespace, weight_dtype, device="cpu", unet_use_linear_projection_in_v2=False):
    import library.model_util as model_util
    from library.original_unet import UNet2DConditionModel

    name_or_path = args.pretrained_model_name_or_path
    name_or_path = os.path.realpath(name_or_path) if os.path.islink(
        name_or_path) else name_or_path
    load_stable_diffusion_format = False
    if True:
        try:
            pipe = StableDiffusionPipeline.from_single_file(
                name_or_path, local_files_only=True, safety_checker=None)
        except EnvironmentError as ex:
            print(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex

        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        global clip_large_tokenizer
        clip_large_tokenizer = pipe.tokenizer
        del pipe

        original_unet = UNet2DConditionModel(
            unet.config.sample_size,
            unet.config.attention_head_dim,
            unet.config.cross_attention_dim,
            unet.config.use_linear_projection,
            unet.config.upcast_attention,
        )
        original_unet.load_state_dict(unet.state_dict())
        unet = original_unet
        print("U-Net converted to original U-Net")

    if args.vae is not None:
        vae = model_util.load_vae(args.vae, weight_dtype)
        print("additional VAE loaded")

    return text_encoder, vae, unet, load_stable_diffusion_format

def generate_image(pipe_class, cmd_args, accelerator, vae, tokenizer, text_encoder, unet, epoch, prompt_dict_list, **kwargs):
    if pipe_class is None:
        print("pipe_class is None")
        return
    import library.train_util

    distributed_state = library.train_util.PartialState()
    org_vae_device = vae.device
    vae.to(distributed_state.device)
    unet = accelerator.unwrap_model(unet)

    if isinstance(text_encoder, (list, tuple)):
        text_encoder = [accelerator.unwrap_model(te) for te in text_encoder]
    else:
        text_encoder = accelerator.unwrap_model(text_encoder)

    default_scheduler = library.train_util.get_my_scheduler(
        sample_sampler="k_euler",
        v_parameterization=cmd_args.v_parameterization,
    )

    pipeline = pipe_class(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=default_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        clip_skip=cmd_args.clip_skip,
    )
    pipeline.to(distributed_state.device)

    workspaces_dir = os.path.dirname(cmd_args.dataset_config)
    sample_images_path = os.path.join(
        workspaces_dir, "sample_images")

    os.makedirs(sample_images_path, exist_ok=True)

    save_dir = sample_images_path
    prompt_replacement = None
    steps = 0

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    image_counter = 0

    with torch.no_grad():
        for prompt_dict in prompt_dict_list:
            if image_counter == 0:
                custom_name = "Sanity Check"
            else:
                custom_name = f"Epoch {image_counter}"

            library.train_util.sample_image_inference(
                accelerator, cmd_args, pipeline, save_dir, prompt_dict, epoch, steps, prompt_replacement
            )

            old_name = f"{cmd_args.output_name}_{epoch:06d}-{steps:06d}_{prompt_dict.get('seed', 0)}.png"
            new_name = f"{custom_name}_{prompt_dict.get('seed', 0)}.png"
            old_path = os.path.join(save_dir, old_name)
            new_path = os.path.join(save_dir, new_name)

            if os.path.exists(old_path):
                os.rename(old_path, new_path)

            image_counter += 1

    del pipeline
    library.train_util.clean_memory_on_device(accelerator.device)

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    vae.to(org_vae_device)