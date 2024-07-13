

import argparse
import hashlib
import os
import shutil
import sys
import json
import subprocess
import time

import torch

from .FL_train_utils import Utils
import folder_paths
import nodes


git_accelerate_urls = {
    "githubfast": "githubfast.com",
    "521github": "521github.com",
    "kkgithub": "kkgithub.com",
}


def FL_KohyaSSCloneRepo_call(args={}):
    FL_dir = Utils.get_minus_zone_models_path()

    branch_repoid = args.get("branch_repoid", "kohya-ss/sd-scripts")
    branch_local_name = args.get("branch_local_name", "kohya_ss_lora")

    git_url = f"https://github.com/{branch_repoid}"
    source = args.get("source", "github")
    kohya_ss_lora_dir = os.path.join(FL_dir, "train_tools", branch_local_name)
    if git_accelerate_urls.get(source, None) is not None:
        git_url = f"https://{git_accelerate_urls[source]}/{branch_repoid}"
    try:
        if not os.path.exists(kohya_ss_lora_dir) or not os.path.exists(os.path.join(kohya_ss_lora_dir, ".git")):
            subprocess.run(
                ["git", "clone", "--depth", "1", git_url, kohya_ss_lora_dir], check=True)

        # git remote set-branches origin 'main'
        branch = args.get("branch", "main")

        short_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=kohya_ss_lora_dir, stdout=subprocess.PIPE, check=True)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=kohya_ss_lora_dir, stdout=subprocess.PIPE, check=True)

        short_current_branch = short_result.stdout.decode().strip()
        long_current_branch = result.stdout.decode().strip()
        print(
            f"(current branch): {long_current_branch}({short_current_branch})")
        print(f"(target branch): {branch}")

        if branch != long_current_branch and branch != short_current_branch:
            subprocess.run(
                ["git", "remote", "set-branches", "origin", branch], cwd=kohya_ss_lora_dir, check=True)
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", branch], cwd=kohya_ss_lora_dir, check=True)

            subprocess.run(
                ["git", "checkout", "."], cwd=kohya_ss_lora_dir, check=True)

            subprocess.run(
                ["git", "checkout", branch], cwd=kohya_ss_lora_dir, check=True)

    except Exception as e:
        raise Exception(f"kohya-ss/sd-scripts...")



def FL_KohyaSSInitWorkspace_call(args={}):
    FL_KohyaSSCloneRepo_call(args)

    workspace_name = args.get("lora_name", None)
    workspace_name = workspace_name.strip()

    if workspace_name is None or workspace_name == "":
        raise Exception("lora(lora_name is required)")

    args["workspace_name"] = workspace_name
    workspaces_dir = os.path.join(
        folder_paths.output_directory, "FL_train_workspaces")

    os.makedirs(workspaces_dir, exist_ok=True)

    workspace_dir = os.path.join(workspaces_dir, workspace_name)
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)

    return (
        args,
    )


def FL_ImageSelecter_call(args={}):
    images = args.get("images")
    pil_images = Utils.tensors2pil_list(images)

    conditioning_images = args.get("conditioning_images", None)
    conditioning_pil_images = None
    if conditioning_images is not None:
        conditioning_pil_images = Utils.tensors2pil_list(conditioning_images)

    resolution = args.get("resolution", 512)

    workspace_config = args.get("workspace_config", {})
    workspace_name = workspace_config.get("workspace_name", None)

    if workspace_name is None or workspace_name == "":
        raise Exception("lora(lora_name is required)")

    workspace_dir = os.path.join(
        folder_paths.output_directory, "FL_train_workspaces", workspace_name)
    train_images_dir = os.path.join(workspace_dir, "train_images")
    os.makedirs(train_images_dir, exist_ok=True)

    conditioning_images_dir = os.path.join(workspace_dir, "conditioning_images")

    force_clear = args.get("force_clear") == "enable"
    force_clear_only_images = args.get("force_clear_only_images") == "enable"
    if force_clear:
        if force_clear_only_images:
            images_files = Utils.listdir(train_images_dir)
            for file in images_files:
                if file.lower().endswith((".png", ".jpg", ".webp", ".caption", ".txt")):
                    os.remove(os.path.join(train_images_dir, file))
        else:
            shutil.rmtree(train_images_dir)
            os.makedirs(train_images_dir, exist_ok=True)

    image_format = args.get("image_format")
    file_extension = "." + image_format
    caption_extension = args.get("caption_extension", ".caption")

    captions = args.get("captions", [])

    saved_images_path = []
    for i, pil_image in enumerate(pil_images):
        pil_image = Utils.resize_max(pil_image, resolution, resolution)
        width, height = pil_image.size
        filename = hashlib.md5(pil_image.tobytes()).hexdigest() + file_extension
        pil_image.save(os.path.join(train_images_dir, filename))
        saved_images_path.append(filename)

        if i < len(captions):
            caption_filename = os.path.splitext(filename)[0] + caption_extension
            with open(os.path.join(train_images_dir, caption_filename), "w", encoding="utf-8") as f:
                f.write(captions[i])

        if conditioning_pil_images is not None:
            os.makedirs(conditioning_images_dir, exist_ok=True)
            conditioning_pil_images[i].resize((width, height)).save(
                os.path.join(conditioning_images_dir, filename))

    if conditioning_images is None:
        conditioning_images_dir = None

    if os.path.exists(os.path.join(workspace_dir, "dataset.json")):
        os.remove(os.path.join(workspace_dir, "dataset.json"))

    if os.path.exists(os.path.join(workspace_dir, "dataset.toml")):
        os.remove(os.path.join(workspace_dir, "dataset.toml"))

    dataset_config_extension = args.get("dataset_config_extension")
    generate_dataset_config(
        os.path.join(workspace_dir, "dataset" + dataset_config_extension),
        enable_bucket=args.get("enable_bucket") == "enable",
        resolution=args.get("resolution"),
        batch_size=args.get("batch_size"),
        image_dir=train_images_dir,
        conditioning_data_dir=conditioning_images_dir,
        caption_extension=caption_extension,
        num_repeats=args.get("num_repeats"),
    )
    return (train_images_dir,)


def FL_KohyaSSUseConfig_call(args={}):
    args = args.copy()
    workspace_config = args.get("workspace_config", {})
    workspace_name = workspace_config.get("workspace_name", None)

    if workspace_name is None or workspace_name == "":
        raise Exception("(workspace_name is required)")

    workspace_dir = os.path.join(
        folder_paths.output_directory, "FL_train_workspaces", workspace_name)

    if not os.path.exists(workspace_dir):
        raise Exception(f": {workspace_dir}")

    workspace_config_file = os.path.join(workspace_dir, "config.json")

    train_config_template = args.get("train_config_template", None)
    train_config_template_dir = args.get("train_config_template_dir", None)
    train_config_template_file = os.path.join(
        train_config_template_dir, train_config_template + ".json")

    # if not os.path.exists(workspace_config_file):
    #     train_config_template_dir = args.get("train_config_template_dir", None)
    #     train_config_template_file = os.path.join(
    #         train_config_template_dir, train_config_template + ".json")
    #     shutil.copy(train_config_template_file, workspace_config_file)

    config = None
    with open(train_config_template_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        config["metadata"]["train_type"] = train_config_template
        ckpt_name = args.get("ckpt_name", "")
        if ckpt_name != "" and ckpt_name is not None:
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            config["train_config"]["pretrained_model_name_or_path"] = ckpt_path

        # output_dir
        output_dir = os.path.join(workspace_dir, "output")
        config["train_config"]["output_dir"] = output_dir

        datetime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # output_name
        config["train_config"]["output_name"] = f"{workspace_name}_{train_config_template}_{datetime}"

        dataset_config_path = os.path.join(
            workspace_dir, "dataset.json")
        if not os.path.exists(dataset_config_path):
            dataset_config_path = os.path.join(
                workspace_dir, "dataset.toml")

        config["train_config"]["dataset_config"] = dataset_config_path

        config["train_config"]["max_train_steps"] = str(
            args.get("max_train_steps"))

        config["train_config"]["max_train_epochs"] = str(
            args.get("max_train_epochs"))
        if config["train_config"]["max_train_epochs"] == "0":
            config["train_config"]["max_train_epochs"] = False

        config["train_config"]["save_every_n_epochs"] = str(
            args.get("save_every_n_epochs"))

        config["train_config"]["learning_rate"] = str(
            args.get("learning_rate"))

        advanced_config = args.get("save_advanced_config", {}).copy()
        if len(advanced_config) == 0:
            advanced_config = args.get("advanced_config", {}).copy()

        for k in advanced_config:
            print(f"{k}= {advanced_config[k]}")

            if type(advanced_config[k]) == str and advanced_config[k] == "":
                if k in config["train_config"]:
                    del config["train_config"][k]
                continue
            elif advanced_config[k] == "enable":
                advanced_config[k] = True
            elif advanced_config[k] == "disable":
                advanced_config[k] = False
            else:
                advanced_config[k] = str(advanced_config[k])
            config["train_config"][k] = advanced_config[k]

        # raise Exception(f"args: {json.dumps(config, indent=4)}")

    if config is None:
        raise Exception(f": {workspace_config_file}")

    with open(workspace_config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # raise Exception(f"FL_KohyaSSUseConfig_call: {args}")
    return (
        args,
    )


def FL_KohyaSSAdvConfig_call(args={}):
    return (
        args,
    )


def config2args(train_parser: argparse.ArgumentParser, config):
    try:
        config_args_list = []
        for key, value in config.items():
            if value is None:
                continue
            if type(value) == bool:
                if value:
                    config_args_list.append(f"--{key}")
            else:
                config_args_list.append(f"--{key}")
                config_args_list.append(str(value))
        args = train_parser.parse_args(config_args_list)
        return args
    except Exception as e:
        raise Exception(f"config2args: {e}")


def check_install():
    try:
        import toml
    except ImportError:
        os.system(f"{sys.executable} -m pip install toml")

    # imagesize
    try:
        import imagesize
    except ImportError:
        os.system(f"{sys.executable} -m pip install imagesize")

    # voluptuous
    try:
        import voluptuous
    except ImportError:
        os.system(f"{sys.executable} -m pip install voluptuous")

    try:
        import diffusers
    except ImportError:
        os.system(f"{sys.executable} -m pip install diffusers")
    try:
        import accelerate
    except ImportError:
        os.system(f"{sys.executable} -m pip install accelerate")


import logging


def generate_dataset_config(output_path, enable_bucket=True, resolution=512, batch_size=1, image_dir=None, conditioning_data_dir=None, caption_extension=".caption", num_repeats=10, ):

    config = {
        'general': {
            'enable_bucket': enable_bucket,
        },
        'datasets': [
            {
                'resolution': resolution,
                'batch_size': batch_size,
                'subsets': [
                    {
                        'image_dir': image_dir,
                        'caption_extension': caption_extension,
                        'num_repeats': num_repeats,
                    },
                ],
            },
        ],
    }

    if conditioning_data_dir is not None:
        config["datasets"][0]["subsets"][0]["conditioning_data_dir"] = conditioning_data_dir

    if output_path.endswith(".toml"):
        check_install()
        import toml
        with open(output_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
    elif output_path.endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    else:
        raise Exception(f": {output_path}")


from PIL import Image


def get_sample_images(workspace_dir, output_name):
    sample_images_dir = os.path.join(
        workspace_dir, "sample_images"
    )
    pil_images = []
    pre_render_texts_x = []
    if os.path.exists(sample_images_dir):
        image_files = Utils.listdir(sample_images_dir)
        image_files = list(
            filter(lambda x: x.lower().endswith(".png"), image_files))
        # output_name
        image_files = list(
            filter(lambda x: x.startswith(output_name), image_files))

        image_files = sorted(image_files, key=lambda x: x)

        for image_file in image_files:
            pil_image = Image.open(os.path.join(sample_images_dir, image_file))
            pil_images.append([pil_image])
            pre_render_texts_x.append(image_file)
    if pil_images is None or len(pil_images) == 0:
        return Image.new("RGB", (512, 512), (255, 255, 255))
    result = Utils.xy_image(
        pre_render_images=pil_images,
        pre_render_texts_x=pre_render_texts_x,
        pre_render_texts_y=[""],
    )
    return result


def run_hook_kohya_ss_run_file(workspace_dir, output_name, kohya_ss_tool_dir, trainer_func, use_screen=False):

    train_config_file = os.path.join(workspace_dir, "config.json")

    exec_pyfile = os.path.join(os.path.dirname(
        __file__), "hook_kohya_ss_run.py",)

    is_running = True

    taesd_type = "sd1_5"
    if trainer_func.find("sd1_5") != -1:
        taesd_type = "sd1_5"
    if trainer_func.find("sdxl") != -1:
        taesd_type = "sdxl"
    if trainer_func.find("hunyuan1_1") != -1:
        taesd_type = "sdxl"

    pb = Utils.progress_bar(0, taesd_type)

    import traceback

    import comfy.model_management

    stop_server = None

    def log_callback(log):
        try:
            comfy.model_management.throw_exception_if_processing_interrupted()
        except Exception as e:
            stop_server()
            if process_instance is not None:
                process_instance.stop()
            return is_running

        try:
            resp = log
            if resp.get("type") == "sample_images":
                global_step = resp.get("global_step")
                xy_img = get_sample_images(workspace_dir, output_name)

                max_side = max(xy_img.width, xy_img.height)
                # print(f"global_step: {global_step}, max_train_steps: {max_train_steps}")

                total_steps = resp.get("total_steps")
                pb.update(
                    int(global_step), int(total_steps), ("JPEG", xy_img, max_side))
            else:
                print(f"LOG: {log}")
        except Exception as e:
            print(f"LOG: {log} e: {e} ")
            print(f"stack: {traceback.format_exc()}")
        return is_running

    stop_server, port = Utils.Simple_Server(log_callback)
    try:
        cmd_list = [sys.executable, exec_pyfile, "--sys_path", kohya_ss_tool_dir,
                    "--config", train_config_file, "--train_func", trainer_func, "--master_port", str(port)]
        startup_script_path_sh = os.path.join(workspace_dir, "start_train.sh")
        startup_script_path_bat = os.path.join(
            workspace_dir, "start_train.bat")
        with open(startup_script_path_sh, "w", encoding="utf-8") as f:
            f.write(" ".join(cmd_list))
        with open(startup_script_path_bat, "w", encoding="utf-8") as f:
            f.write(" ".join(cmd_list))

        from .FL_train_utils import HSubprocess

        screen_name = None
        if use_screen:
            screen_name = "FL_train_tools_core"

        process_instance = HSubprocess(
            cmd_list, screen_name=screen_name)
        process_instance.wait()

        stop_server()
        is_running = False
    except Exception as e:
        stop_server()
        is_running = False
        raise Exception(f"")


def generate_kohya_ss_config(args):
    args = args.copy()
    workspace_config = args.get("workspace_config", {}).copy()
    advanced_config = args.get("advanced_config", {}).copy()
    train_config = args.get("train_config", {}).copy()

    workspace_name = workspace_config.get("workspace_name", None)
    if workspace_name is None or workspace_name == "":
        raise Exception("(workspace_name is required)")
    workspace_dir = os.path.join(
        folder_paths.output_directory, "FL_train_workspaces", workspace_name)

    if not os.path.exists(workspace_dir):
        raise Exception(f": {workspace_dir}")

    workspace_config_file = os.path.join(workspace_dir, "config.json")

    train_config_template = args.get("train_config_template", None)
    train_config_template_dir = os.path.join(
        os.path.dirname(__file__), "configs", "kohya_ss_lora")
    train_config_template_file = os.path.join(
        train_config_template_dir, train_config_template + ".json")

    # raise Exception(f"args: {json.dumps(args, indent=4, ensure_ascii=False)}")
    config = None
    with open(train_config_template_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        config["metadata"]["train_type"] = train_config_template
        ckpt_name = args.get("ckpt_name", "")
        if ckpt_name != "" and ckpt_name is not None:
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            config["train_config"]["pretrained_model_name_or_path"] = ckpt_path

        # output_dir
        output_dir = os.path.join(workspace_dir, "output")
        config["train_config"]["output_dir"] = output_dir

        datetime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # output_name
        config["train_config"]["output_name"] = f"{workspace_name}_{train_config_template}_{datetime}"

        dataset_config_path = os.path.join(
            workspace_dir, "dataset.json")
        if not os.path.exists(dataset_config_path):
            dataset_config_path = os.path.join(
                workspace_dir, "dataset.toml")

        config["train_config"]["dataset_config"] = dataset_config_path

        config["train_config"]["max_train_steps"] = str(
            args.get("max_train_steps"))

        config["train_config"]["max_train_epochs"] = str(
            args.get("max_train_epochs"))
        if config["train_config"]["max_train_epochs"] == "0":
            config["train_config"]["max_train_epochs"] = False

        config["train_config"]["save_every_n_epochs"] = str(
            args.get("save_every_n_epochs"))

        config["train_config"]["learning_rate"] = str(
            args.get("learning_rate"))

        for k in advanced_config:
            print(f"{k}= {advanced_config[k]}")

            if type(advanced_config[k]) == str and advanced_config[k] == "":
                if k in config["train_config"]:
                    del config["train_config"][k]
                continue
            elif advanced_config[k] == "enable":
                advanced_config[k] = True
            elif advanced_config[k] == "disable":
                advanced_config[k] = False
            else:
                advanced_config[k] = str(advanced_config[k])
            config["train_config"][k] = advanced_config[k]

        # raise Exception(f"args: {json.dumps(config, indent=4)}")

    if config is None:
        raise Exception(f": {workspace_config_file}")

    # raise Exception(f"FL_KohyaSSUseConfig_call: {args}")
    return config


def FL_KohyaSSTrain_call(args={}):
    args = args.copy()
    workspace_config = args.get("workspace_config").copy()
    base_lora = args.get("base_lora", "empty")
    sample_generate = args.get("sample_generate", "enable")
    sample_prompt = args.get("sample_prompt", "")

    workspace_name = workspace_config.get("workspace_name")
    workspace_dir = os.path.join(
        folder_paths.output_directory, "FL_train_workspaces", workspace_name)

    if not os.path.exists(workspace_dir):
        raise Exception(f": {workspace_dir}")

    config = generate_kohya_ss_config(args)

    branch_local_name = workspace_config.get(
        "branch_local_name", "kohya_ss_lora")
    kohya_ss_tool_dir = os.path.join(
        Utils.get_minus_zone_models_path(), "train_tools", branch_local_name)

    if kohya_ss_tool_dir not in sys.path:
        sys.path.append(kohya_ss_tool_dir)
    check_install()

    if base_lora == "empty":
        pass
    elif base_lora == "latest":
        workspace_lora_dir = os.path.join(workspace_dir, "output")
        if os.path.exists(workspace_lora_dir):
            workspace_lora_files = Utils.listdir(workspace_lora_dir)
            workspace_lora_files = list(
                filter(lambda x: x.endswith(".safetensors"), workspace_lora_files))
            workspace_lora_files = list(
                map(lambda x: os.path.join(workspace_lora_dir, x), workspace_lora_files))
            # 排序
            workspace_lora_files = sorted(
                workspace_lora_files, key=lambda x: os.path.getctime(x), reverse=True)
            if len(workspace_lora_files) > 0:
                base_lora = os.path.join(
                    workspace_lora_dir, workspace_lora_files[0])
        else:
            base_lora = "empty"
    else:
        pass

    if base_lora != "empty" and os.path.exists(base_lora):
        config["train_config"]["network_weights"] = base_lora
        config["train_config"]["dim_from_weights"] = True

        if "network_dim" in config["train_config"]:
            del config["train_config"]["network_dim"]
        if "network_alpha" in config["train_config"]:
            del config["train_config"]["network_alpha"]
        if "network_dropout" in config["train_config"]:
            del config["train_config"]["network_dropout"]

    base_controlnet = args.get("base_controlnet", "empty")
    if base_controlnet == "empty":
        pass
    elif base_controlnet == "latest":
        workspace_controlnet_dir = os.path.join(workspace_dir, "output")
        if os.path.exists(workspace_controlnet_dir):
            workspace_controlnet_files = Utils.listdir(
                workspace_controlnet_dir)
            workspace_controlnet_files = list(
                filter(lambda x: x.endswith(".safetensors"), workspace_controlnet_files))
            workspace_controlnet_files = list(
                map(lambda x: os.path.join(workspace_controlnet_dir, x), workspace_controlnet_files))
            # 排序
            workspace_controlnet_files = sorted(
                workspace_controlnet_files, key=lambda x: os.path.getctime(x), reverse=True)
            if len(workspace_controlnet_files) > 0:
                base_controlnet = os.path.join(
                    workspace_controlnet_dir, workspace_controlnet_files[0])
        else:
            base_controlnet = "empty"
    else:
        pass

    if base_controlnet != "empty" and os.path.exists(base_controlnet):
        config["train_config"]["controlnet_model_name_or_path"] = base_controlnet

    train_type = config.get("metadata").get("train_type")

    if sample_generate == "enable":
        config["other_config"] = {
            "sample_prompt": sample_prompt,
        }
    else:
        config["other_config"] = {}

    workspace_config_file = os.path.join(workspace_dir, "config.json")
    with open(workspace_config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    # raise Exception(
    #     f"config: {json.dumps(config, indent=4, ensure_ascii=False)}")

    output_name = config["train_config"].get("output_name")

    if train_type == "lora_sd1_5":
        run_hook_kohya_ss_run_file(
            workspace_dir, output_name, kohya_ss_tool_dir, "run_lora_sd1_5")
    elif train_type == "lora_sdxl":
        run_hook_kohya_ss_run_file(
            workspace_dir, output_name, kohya_ss_tool_dir, "run_lora_sdxl")
    elif train_type == "controlnet_sd1_5":

        conditioning_images_dir = os.path.join(
            workspace_dir, "conditioning_images")
        conditioning_images_onec = ""
        if os.path.exists(conditioning_images_dir):
            conditioning_images_onec = Utils.listdir(
                conditioning_images_dir)[0]
            config["other_config"]["controlnet_image"] = os.path.join(
                conditioning_images_dir, conditioning_images_onec)
            with open(workspace_config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)

        run_hook_kohya_ss_run_file(
            workspace_dir, output_name, kohya_ss_tool_dir, "run_controlnet_sd1_5")
    elif train_type == "lora_hunyuan1_2" or train_type == "lora_hunyuan1_1":
        hunyuan_models_config = args.get(
            "hunyuan_models_config", {})
        from .FL_train_tools_core_HYDiT import check_model_auto_download

        hunyuan_models_config["version"] = config.get(
            "metadata").get("version")
        config["other_config"]["hunyuan_models_config"] = check_model_auto_download(
            hunyuan_models_config)

        with open(workspace_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        run_hook_kohya_ss_run_file(
            workspace_dir, output_name, kohya_ss_tool_dir, "run_lora_hunyuan1_2")
    else:
        raise Exception(
            f": {train_type}")

    return (
        "",
    )


def FL_KohyaSS_KohakuBlueleaf_HYHiDSimpleT2I_call(args={}):
    args = args.copy()
    FL_KohyaSSCloneRepo_call(args)
    from .FL_train_tools_core_HYDiT import check_model_auto_download
    args = check_model_auto_download(args)
    import numpy as np
    import torch
    seed = args.get("seed", 0)
    torch.manual_seed(seed)
    from packaging import version
    from transformers import AutoTokenizer, BertModel
    from diffusers.models import AutoencoderKL
    try:
        from k_diffusion.external import DiscreteVDDPMDenoiser
        from k_diffusion.sampling import sample_euler_ancestral, get_sigmas_exponential, sample_dpmpp_2m_sde
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "k-diffusion"])
        from k_diffusion.external import DiscreteVDDPMDenoiser
        from k_diffusion.sampling import sample_euler_ancestral, get_sigmas_exponential, sample_dpmpp_2m_sde

    branch_local_name = args.get("branch_local_name")
    kohya_ss_tool_dir = os.path.join(
        Utils.get_minus_zone_models_path(), "train_tools", branch_local_name)
    if kohya_ss_tool_dir not in sys.path:
        sys.path.append(kohya_ss_tool_dir)
    from library.hunyuan_models import DiT_g_2, MT5Embedder
    from library.hunyuan_utils import get_cond, calc_rope
    from networks.lora import create_network_from_weights

    def load_scheduler_sigmas(beta_start=0.00085, beta_end=0.018, num_train_timesteps=1000):
        betas = torch.linspace(beta_start**0.5, beta_end **
                               0.5, num_train_timesteps, dtype=torch.float32) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        sigmas = torch.from_numpy(sigmas)
        return alphas_cumprod, sigmas

    version = args.get("version")
    BETA_END = None
    USE_EXTRA_COND = None
    if version == "1.1":
        BETA_END = 0.03
        USE_EXTRA_COND = True
    else:
        BETA_END = 0.018
        USE_EXTRA_COND = False

    ATTN_MODE = "xformers"
    CLIP_TOKENS = 75 * 2 + 2
    dtype = DTYPE = torch.float16
    device = DEVICE = "cuda"

    image = None

    with torch.inference_mode(True), torch.no_grad():
        alphas, sigmas = load_scheduler_sigmas(beta_end=BETA_END)

        tokenizer_path = args.get("tokenizer_path")
        clip_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True)
        clip_tokenizer.eos_token_id = 2

        text_encoder_path = args.get("text_encoder_path")

        clip_encoder = Utils.model_cache_get(
            model_type="HYDiT_clip_encoder", model_path=text_encoder_path,)
        if clip_encoder is None:
            clip_encoder = (
                BertModel.from_pretrained(
                    text_encoder_path, local_files_only=True).to(device).to(dtype)
            )
            Utils.model_cache_set(
                model_type="HYDiT_clip_encoder", model_path=text_encoder_path, model=clip_encoder)

        t5_encoder_path = args.get("t5_encoder_path")
        if t5_encoder_path != "none" and os.path.exists(t5_encoder_path):
            mt5_embedder = Utils.model_cache_get(
                model_type="HYDiT_mt5_embedder", model_path=t5_encoder_path,)
            if mt5_embedder is None:
                mt5_embedder = (
                    MT5Embedder(t5_encoder_path, torch_dtype=dtype,
                                max_length=256,).to(device).to(dtype)
                )
                Utils.model_cache_set(
                    model_type="HYDiT_mt5_embedder", model_path=t5_encoder_path, model=mt5_embedder)
        else:
            from .FL_train_utils import CustomizeMT5Embedder
            mt5_embedder = (
                CustomizeMT5Embedder(
                    batch_size=1,
                )
                .to(device)
                .to(dtype)
            )

        vae_ema_path = args.get("vae_ema_path")
        vae = Utils.model_cache_get(
            model_type="HYDiT_vae", model_path=vae_ema_path,)
        if vae is None:
            vae = (
                AutoencoderKL.from_pretrained(
                    vae_ema_path, local_files_only=True)
                .to(device)
                .to(dtype)
            )
            Utils.model_cache_set(
                model_type="HYDiT_vae", model_path=vae_ema_path, model=vae)

        unet_path = args.get("unet_path")
        lora_path = args.get("lora_path")

        denoiser_args = Utils.model_cache_get(
            model_type="HYDiT_unet_merge_lora", model_path=f"{unet_path}_{lora_path}")

        if denoiser_args is None:
            denoiser, patch_size, head_dim = DiT_g_2(
                input_size=(128, 128), use_extra_cond=USE_EXTRA_COND)
            state_dict = torch.load(unet_path)
            denoiser.load_state_dict(state_dict)
            denoiser.to(device).to(dtype)
            denoiser.eval()
            denoiser.disable_fp32_silu()
            denoiser.disable_fp32_layer_norm()
            denoiser.set_attn_mode(ATTN_MODE)

            if lora_path is not None and lora_path != "none":
                if not os.path.exists(lora_path):
                    raise Exception(f"lora_path: {lora_path} 不存在")
                lora_net, state_dict = create_network_from_weights(
                    multiplier=1.0,
                    file=lora_path,
                    vae=vae,
                    text_encoder=[clip_encoder, mt5_embedder],
                    unet=denoiser,
                )
                lora_net.apply_to(
                    text_encoder=[clip_encoder, mt5_embedder],
                    unet=denoiser,
                )
                lora_net.load_state_dict(state_dict)
                lora_net = lora_net.to(DEVICE, dtype=DTYPE)

            Utils.model_cache_set(
                model_type="HYDiT_unet_merge_lora", model_path=f"{unet_path}_{lora_path}", model=(denoiser, patch_size, head_dim))
        else:
            denoiser, patch_size, head_dim = denoiser_args

        vae.requires_grad_(False)
        mt5_embedder.to(torch.float16)
        prompt = args.get("prompt")
        negative_prompt = args.get("negative_prompt")
        with torch.autocast("cuda"):
            clip_h, clip_m, mt5_h, mt5_m = get_cond(
                prompt,
                mt5_embedder,
                clip_tokenizer,
                clip_encoder,
                # Should be same as original implementation with max_length_clip=77
                # Support 75*n + 2
                max_length_clip=CLIP_TOKENS,
            )
            neg_clip_h, neg_clip_m, neg_mt5_h, neg_mt5_m = get_cond(
                negative_prompt,
                mt5_embedder,
                clip_tokenizer,
                clip_encoder,
                max_length_clip=CLIP_TOKENS,
            )
            clip_h = torch.concat([clip_h, neg_clip_h], dim=0)
            clip_m = torch.concat([clip_m, neg_clip_m], dim=0)
            mt5_h = torch.concat([mt5_h, neg_mt5_h], dim=0)
            mt5_m = torch.concat([mt5_m, neg_mt5_m], dim=0)
            torch.cuda.empty_cache()

        style = torch.as_tensor([0] * 2, device=DEVICE)
        W = args.get("width")
        H = args.get("height")

        size_cond = [H, W, H, W, 0, 0]
        image_meta_size = torch.as_tensor([size_cond] * 2, device=DEVICE)
        freqs_cis_img = calc_rope(H, W, patch_size, head_dim)

        denoiser_wrapper = DiscreteVDDPMDenoiser(
            # A quick patch for learn_sigma
            lambda *args, **kwargs: denoiser(* \
                                             args, **kwargs).chunk(2, dim=1)[0],
            alphas,
            False,
        ).to(DEVICE)

        CFG_SCALE = cfg = args.get("cfg", 5.0)
        STEPS = steps = args.get("steps", 25)

        def cfg_denoise_func(x, sigma):
            cond, uncond = denoiser_wrapper(
                x.repeat(2, 1, 1, 1),
                sigma.repeat(2),
                encoder_hidden_states=clip_h,
                text_embedding_mask=clip_m,
                encoder_hidden_states_t5=mt5_h,
                text_embedding_mask_t5=mt5_m,
                image_meta_size=image_meta_size,
                style=style,
                cos_cis_img=freqs_cis_img[0],
                sin_cis_img=freqs_cis_img[1],
            ).chunk(2, dim=0)
            return uncond + (cond - uncond) * CFG_SCALE

        sigmas = denoiser_wrapper.get_sigmas(STEPS).to(DEVICE)
        sigmas = get_sigmas_exponential(
            STEPS, denoiser_wrapper.sigma_min, denoiser_wrapper.sigma_max, DEVICE
        )
        x1 = torch.randn(1, 4, H // 8, W // 8,
                         dtype=torch.float16, device=DEVICE)

        pbar = Utils.progress_bar(STEPS, "sdxl")
        preview = pbar.get_previewer()

        def generate_callback(args):
            try:
                i = args.get("i")
                latents = args.get("denoised")
                # decode_latent_to_preview_image
                if hasattr(preview, "decode_latent_to_preview_image"):
                    pil_img = preview.decode_latent_to_preview_image(
                        None,
                        latents,
                    )[1]
                else:
                    pil_img = None
                pbar.update(i, STEPS, pil_img)
            except Exception as e:
                print("generate_callback error:", e)
                raise e

        with torch.autocast("cuda"):
            scheduler = args.get("scheduler")
            if scheduler == "euler_ancestral":
                sample = sample_euler_ancestral(
                    cfg_denoise_func,
                    x1 * sigmas[0],
                    sigmas,
                    callback=generate_callback,
                )
            else:
                sample = sample_dpmpp_2m_sde(
                    cfg_denoise_func,
                    x1 * sigmas[0],
                    sigmas,
                    callback=generate_callback,
                )
            torch.cuda.empty_cache()
            with torch.no_grad():
                latent = sample / 0.13025
                image = vae.decode(latent).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.permute(0, 2, 3, 1)

    keep_device = args.get("keep_device", "enable")
    if keep_device == "disable":
        Utils.model_cache_clean(
            model_type="HYDiT_clip_encoder")
        Utils.model_cache_clean(
            model_type="HYDiT_mt5_embedder")
        Utils.model_cache_clean(
            model_type="HYDiT_vae")
        Utils.model_cache_clean(
            model_type="HYDiT_unet_merge_lora")

    return (image,)


def FL_TrainToolsDebug_call(kwargs):
    from pprint import pprint, pformat
    object = kwargs["object"]
    indent = kwargs["indent"]
    depth = kwargs["depth"]
    width = kwargs["width"]
    compact = kwargs["compact"] == "enable"
    sort_keys = kwargs["sort_keys"] == "enable"
    underscore_numbers = kwargs["underscore_numbers"] == "enable"

    index = kwargs["index"]
    # 用点号分割
    index = index.split(".")
    for i in index:
        if i == "":
            continue
        if isinstance(object, (list, tuple)):
            object = object[int(i)]
        elif isinstance(object, dict):
            object = object.get(i)
        elif hasattr(object, i):
            object = getattr(object, i)
        else:
            object = object[i]

    debug = pformat(object, indent=indent, depth=depth, width=width,
                    compact=compact, sort_dicts=sort_keys, underscore_numbers=underscore_numbers)
    return (debug,)
