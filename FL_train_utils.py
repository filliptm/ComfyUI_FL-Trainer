
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Tuple
import warnings
import numpy as np
import folder_paths
import base64
from PIL import Image, ImageFilter
import io
import torch
import re
import hashlib
import cv2
# sys.path.append(os.path.join(os.path.dirname(__file__)))
temp_directory = folder_paths.get_temp_directory()
from tqdm import tqdm
import requests
import comfy.utils


CACHE_POOL = {}


class Utils:
    def Md5(str):
        return hashlib.md5(str.encode('utf-8')).hexdigest()

    def check_frames_path(frames_path):

        if frames_path == "" or frames_path.startswith(".") or frames_path.startswith("/") or frames_path.endswith("/") or frames_path.endswith("\\"):
            return "frames_path"

        frames_path = os.path.join(
            folder_paths.get_output_directory(), frames_path)

        if frames_path == folder_paths.get_output_directory():
            return "frames_path"

        return ""

    def base64_to_pil_image(base64_str):
        if base64_str is None:
            return None
        if len(base64_str) == 0:
            return None
        if type(base64_str) not in [str, bytes]:
            return None
        if base64_str.startswith("data:image/png;base64,"):
            base64_str = base64_str.split(",")[-1]
        base64_str = base64_str.encode("utf-8")
        base64_str = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(base64_str))

    def pil_image_to_base64(pil_image):
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = str(img_str, encoding="utf-8")
        return f"data:image/png;base64,{img_str}"

    def listdir_png(path):
        try:
            files = Utils.listdir(path)
            new_files = []
            for file in files:
                if file.lower().endswith(".png"):
                    new_files.append(file)
            files = new_files
            files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
            return files
        except Exception as e:
            return []

    def listdir(path):
        try:
            files = os.listdir(path)
            # 排除.开头的文件
            files = [file for file in files if not file.startswith(".")]
            return files
        except Exception as e:
            return []

    def tensor2pil(image):
        return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def tensors2pil_list(images):
        return [Utils.tensor2pil(image) for image in images]

    # Convert PIL to Tensor

    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)[0]

    def pil2cv(image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    def cv2pil(image):
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def list_tensor2tensor(data):
        result_tensor = torch.stack(data)
        return result_tensor

    def loadImage(path):
        img = Image.open(path)
        img = img.convert("RGB")
        return img

    def vae_encode_crop_pixels(pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def native_vae_encode(vae, image):
        pixels = Utils.vae_encode_crop_pixels(image)
        t = vae.encode(pixels[:, :, :, :3])
        return {"samples": t}

    def native_vae_encode_for_inpaint(vae, pixels, mask):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape(
            (-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        pixels = pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        # grow mask by a few pixels to keep things seamless in latent space

        mask_erosion = mask

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5
        t = vae.encode(pixels)

        return {"samples": t, "noise_mask": (mask_erosion[:, :, :x, :y].round())}

    def native_vae_decode(vae, samples):
        return vae.decode(samples["samples"])

    def native_clip_text_encode(clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def cache_get(key):
        return CACHE_POOL.get(key, None)

    def cache_set(key, value):
        global CACHE_POOL
        CACHE_POOL[key] = value
        return True

    def model_cache_clean(model_type):
        global CACHE_POOL
        for key in list(CACHE_POOL.keys()):
            if key.startswith(f"model_cache_{model_type}"):
                del CACHE_POOL[key]
        torch.cuda.empty_cache()
        return True

    def model_cache_get(model_type, model_path):
        resp = Utils.cache_get(f"model_cache_{model_type}")
        if resp is None:
            return None
        cache_model_path = resp.get("model_path")
        if cache_model_path != model_path:
            Utils.model_cache_clean(f"model_cache_{model_type}")
            return None
        return resp.get("model")

    def model_cache_set(model_type, model_path, model):
        key = f"model_cache_{model_type}"
        if Utils.cache_get(key) is not None:
            Utils.model_cache_clean(key)
        Utils.cache_set(key, {"model_path": model_path, "model": model})
        return True

    def get_FL_models_path():
        models_path = os.path.join(
            folder_paths.models_dir, "FL_Kohya")
        os.makedirs(models_path, exist_ok=True)
        return models_path

    def get_comfyui_models_path():
        return folder_paths.models_dir

    def translate_text(text, from_code, to_code):
        try:
            import argostranslate
            from argostranslate import translate
        except ImportError:
            subprocess.run([
                sys.executable, "-m",
                "pip", "install", "argostranslate"], check=True)
            import argostranslate
            from argostranslate import translate

        try:
            translation = translate.get_translation_from_codes(
                from_code, to_code)
            if translation is None:
                raise Exception("Translation not found")

        except Exception as e:
            print(e)
            argostranslate.package.update_package_index()
            available_packages = argostranslate.package.get_available_packages()
            package_to_install = next(
                filter(
                    lambda x: (x.from_code == from_code and x.to_code ==
                               to_code), available_packages,
                )
            )
            download_path = package_to_install.download()
            print("package_to_install.download():", download_path)
            argostranslate.package.install_from_path(download_path)

            translation = translate.get_translation_from_codes(
                from_code, to_code)
            if translation is None:
                return text

        # Translate
        translatedText = translation.translate(
            text)

        return translatedText

    def zh2en(text):
        return Utils.translate_text(text, "zh", "en")

    def en2zh(text):
        return Utils.translate_text(text, "en", "zh")

    def prompt_zh_to_en(prompt):
        prompt = prompt.replace("，", ",")
        prompt = prompt.replace("。", ",")
        prompt = prompt.replace("\n", ",")
        tags = prompt.split(",")
        # 判断是否有中文
        for i, tag in enumerate(tags):
            if re.search(u'[\u4e00-\u9fff]', tag):
                tags[i] = Utils.zh2en(tag)
                # 如果第一个字母是大写,转为小写
                if tags[i][0].isupper():
                    tags[i] = tags[i].lower().replace(".", "")

        return ",".join(tags)

    def mask_resize(mask, width, height):
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = torch.nn.functional.interpolate(
            mask, size=(height, width), mode="bilinear")
        mask = mask.squeeze(0).squeeze(0)
        return mask

    def mask_threshold(interested_mask):
        mask_image = Utils.tensor2pil(interested_mask)
        mask_image_cv2 = Utils.pil2cv(mask_image)
        ret, thresh1 = cv2.threshold(
            mask_image_cv2, 127, 255, cv2.THRESH_BINARY)
        thresh1 = Utils.cv2pil(thresh1)
        thresh1 = np.array(thresh1)
        thresh1 = thresh1[:, :, 0]
        return Utils.pil2tensor(thresh1)

    def mask_erode(interested_mask, value):
        value = int(value)
        mask_image = Utils.tensor2pil(interested_mask)
        mask_image_cv2 = Utils.pil2cv(mask_image)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(mask_image_cv2, kernel, iterations=value)
        erosion = Utils.cv2pil(erosion)
        erosion = np.array(erosion)
        erosion = erosion[:, :, 0]
        return Utils.pil2tensor(erosion)

    def mask_dilate(interested_mask, value):
        value = int(value)
        mask_image = Utils.tensor2pil(interested_mask)
        mask_image_cv2 = Utils.pil2cv(mask_image)
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(mask_image_cv2, kernel, iterations=value)
        dilation = Utils.cv2pil(dilation)
        dilation = np.array(dilation)
        dilation = dilation[:, :, 0]
        return Utils.pil2tensor(dilation)

    def mask_edge_opt(interested_mask, edge_feathering):

        mask_image = Utils.tensor2pil(interested_mask)
        mask_image_cv2 = Utils.pil2cv(mask_image)

        # 高斯模糊
        dilation2 = Utils.cv2pil(mask_image_cv2)
        dilation2 = mask_image.filter(
            ImageFilter.GaussianBlur(edge_feathering))

        # mask_image dilation2 图片蒙版叠加
        dilation2 = Utils.pil2cv(dilation2)
        # dilation2[mask_image_cv2 < 127] = 0
        dilation2 = Utils.cv2pil(dilation2)
        # to RGB
        dilation2 = np.array(dilation2)
        dilation2 = dilation2[:, :, 0]
        return Utils.pil2tensor(dilation2)

    def mask_composite(destination, source, x, y, mask=None, multiplier=8, resize_source=False):
        source = source.to(destination.device)
        if resize_source:
            source = torch.nn.functional.interpolate(source, size=(
                destination.shape[2], destination.shape[3]), mode="bilinear")

        source = comfy.utils.repeat_to_batch_size(source, destination.shape[0])

        x = max(-source.shape[3] * multiplier,
                min(x, destination.shape[3] * multiplier))
        y = max(-source.shape[2] * multiplier,
                min(y, destination.shape[2] * multiplier))

        left, top = (x // multiplier, y // multiplier)
        right, bottom = (left + source.shape[3], top + source.shape[2],)

        if mask is None:
            mask = torch.ones_like(source)
        else:
            mask = mask.to(destination.device, copy=True)
            mask = torch.nn.functional.interpolate(mask.reshape(
                (-1, 1, mask.shape[-2], mask.shape[-1])), size=(source.shape[2], source.shape[3]), mode="bilinear")
            mask = comfy.utils.repeat_to_batch_size(mask, source.shape[0])

        # calculate the bounds of the source that will be overlapping the destination
        # this prevents the source trying to overwrite latent pixels that are out of bounds
        # of the destination
        visible_width, visible_height = (
            destination.shape[3] - left + min(0, x), destination.shape[2] - top + min(0, y),)

        mask = mask[:, :, :visible_height, :visible_width]
        inverse_mask = torch.ones_like(mask) - mask

        source_portion = mask * source[:, :, :visible_height, :visible_width]
        destination_portion = inverse_mask * \
            destination[:, :, top:bottom, left:right]

        destination[:, :, top:bottom,
                    left:right] = source_portion + destination_portion
        return destination

    def latent_upscale_by(samples, scale_by):
        s = samples.copy()
        width = round(samples["samples"].shape[3] * scale_by)
        height = round(samples["samples"].shape[2] * scale_by)
        s["samples"] = comfy.utils.common_upscale(
            samples["samples"], width, height, "nearest-exact", "disabled")
        return s

    def resize_by(image, percent):
        # 判断类型是否为PIL
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        width, height = image.size
        new_width = int(width * percent)
        new_height = int(height * percent)
        return image.resize((new_width, new_height), Image.LANCZOS)

    def resize_max(im, dst_w, dst_h):
        src_w, src_h = im.size

        if src_h > src_w:
            newWidth = dst_w
            newHeight = dst_w * src_h // src_w
        else:
            newWidth = dst_h * src_w // src_h
            newHeight = dst_h

        newHeight = newHeight // 8 * 8
        newWidth = newWidth // 8 * 8

        return im.resize((newWidth, newHeight), Image.Resampling.LANCZOS)

    def resize_min(im, dst_w, dst_h):
        src_w, src_h = im.size

        if src_h < src_w:
            newWidth = dst_w
            newHeight = dst_w * src_h // src_w
        else:
            newWidth = dst_h * src_w // src_h
            newHeight = dst_h

        newHeight = newHeight // 8 * 8
        newWidth = newWidth // 8 * 8

        return im.resize((newWidth, newHeight), Image.Resampling.LANCZOS)

    def add_watermark(image, watermark):
        if watermark == "":
            return image

        try:
            import PIL
            from PIL import ImageDraw, ImageFont
        except ImportError:
            subprocess.run([
                sys.executable, "-m",
                "pip", "install", "Pillow"], check=True)
            import PIL
            from PIL import ImageDraw, ImageFont

        #PIL
        pil_version = PIL.__version__

        if pil_version >= "10.0.0":
            def textsize(self, text, font):
                left, top, right, bottom = self.textbbox((0, 0), text, font)
                return right - left, bottom - top
            ImageDraw.ImageDraw.textsize = textsize

        font_fullpath = Utils.download_model(
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/MinusZoneAIModels/repo?Revision=master&FilePath=font%2FAlibabaPuHuiTi-2-75-SemiBold.ttf",
                "output": "font/AlibabaPuHuiTi-2-75-SemiBold.ttf",
            }
        )

        watermarks = watermark.split("\n")

        width, height = image.size
        short_edge = min(width, height)
        font_size = short_edge // 12

        font = ImageFont.truetype(font_fullpath, font_size)

        # print("pil_version:", pil_version)

        draw = ImageDraw.Draw(image)

        text = watermarks[0]
        textwidth, textheight = draw.textsize(text, font)

        x = (width - textwidth) // 2

        bottom = 10

        y = height - textheight - (textheight * 0.4 + bottom + 8)
        draw.text((x, y), text, font=font)

        if len(watermarks) > 1:
            y1 = y + textheight
            text = watermarks[1]
            font_size = int(font_size * 0.4)
            font = ImageFont.truetype(font_fullpath, font_size)
            textwidth, textheight = draw.textsize(text, font)
            x = (width - textwidth) // 2
            y = y1 - bottom + 4
            draw.text((x, y), text, font=font)

        return image

    def get_device():
        return comfy.model_management.get_torch_device()

    def download_file(url, filepath, threads=8, retries=6):

        get_size_tmp = requests.get(url, stream=True)
        total_size = int(get_size_tmp.headers.get("content-length", 0))

        print(f"Downloading {url} to {filepath} with size {total_size} bytes")

        base_filename = os.path.basename(filepath)
        cache_dir = os.path.join(os.path.dirname(
            filepath), f"{base_filename}.t_{threads}_cache")
        os.makedirs(cache_dir, exist_ok=True)

        def get_total_existing_size():
            fs = os.listdir(cache_dir)
            existing_size = 0
            for f in fs:
                if f.startswith("block_"):
                    existing_size += os.path.getsize(
                        os.path.join(cache_dir, f))
            return existing_size

        total_existing_size = get_total_existing_size()

        if total_size != 0 and total_existing_size != total_size:

            with tqdm(total=total_size, initial=total_existing_size, unit="B", unit_scale=True) as progress_bar:
                all_threads = []

                for i in range(threads):
                    cache_filepath = os.path.join(cache_dir, f"block_{i}")

                    start = total_size // threads * i
                    end = total_size // threads * (i + 1) - 1

                    if i == threads - 1:
                        end = total_size

                    # Check if the file already exists
                    if os.path.exists(cache_filepath):
                        # Get the size of the existing file
                        existing_size = os.path.getsize(cache_filepath)
                    else:
                        existing_size = 0

                    headers = {"Range": f"bytes={start + existing_size}-{end}"}
                    if end == total_size:
                        headers = {"Range": f"bytes={start + existing_size}-"}
                    if start + existing_size >= end:
                        continue
                    # print(f"Downloading {cache_filepath} with headers bytes={start + existing_size}-{end}")

                    # Streaming, so we can iterate over the response.
                    response = requests.get(url, stream=True, headers=headers)

                    def download_file_thread(response, cache_filepath):
                        block_size = 1024
                        if end - (start + existing_size) < block_size:
                            block_size = end - (start + existing_size)
                        with open(cache_filepath, "ab") as file:
                            for data in response.iter_content(block_size):
                                file.write(data)
                                progress_bar.update(
                                    len(data)
                                )

                    t = threading.Thread(
                        target=download_file_thread, args=(response, cache_filepath))

                    all_threads.append(t)

                    t.start()

                for t in all_threads:
                    t.join()

            if total_size != 0 and get_total_existing_size() > total_size:
                # 文件下载失败
                shutil.rmtree(cache_dir)
                raise RuntimeError("Download failed, file is incomplete")

            if total_size != 0 and total_size != get_total_existing_size():
                if retries > 0:
                    retries -= 1
                    print(
                        f"Download failed: {total_size} != {get_total_existing_size()}, retrying... {retries} retries left")
                    return Utils.download_file(url, filepath, threads, retries)

                # 文件损坏
                raise RuntimeError(
                    f"Download failed: {total_size} != {get_total_existing_size()}")

        if os.path.exists(filepath):
            shutil.move(filepath, filepath + ".old." +
                        time.strftime("%Y%m%d%H%M%S"))

        # merge the files
        with open(filepath, "wb") as f:
            for i in range(threads):
                cache_filepath = os.path.join(cache_dir, f"block_{i}")
                with open(cache_filepath, "rb") as cf:
                    f.write(cf.read())

        shutil.rmtree(cache_dir)
        return filepath

    def hf_download_model(url, only_get_path=False):
        if not url.startswith("https://"):
            raise ValueError("URL must start with https://")
        if url.startswith("https://huggingface.co/") or url.startswith("https://hf-mirror.com/"):
            base_model_path = os.path.abspath(os.path.join(
                Utils.get_models_path(), "transformers_models"))
            # https://huggingface.co/FaradayDotDev/llama-3-8b-Instruct-GGUF/resolve/main/llama-3-8b-Instruct.Q2_K.gguf?download=true
            texts = url.split("?")[0].split("/")
            file_name = texts[-1]
            zone_path = f"{texts[3]}/{texts[4]}"

            save_path = os.path.join(base_model_path, zone_path, file_name)

            if os.path.exists(save_path) is False:
                if only_get_path:
                    return None
                os.makedirs(os.path.join(
                    base_model_path, zone_path), exist_ok=True)
                Utils.download_file(url, save_path)

            Utils.print_log(
                f"File {save_path} => {os.path.getsize(save_path)} ")


            if os.path.getsize(save_path) == 0:
                if only_get_path:
                    return None
                os.remove(save_path)
                raise ValueError(f"Download failed: {url}")
            return save_path
        else:
            texts = url.split("?")[0].split("/")
            host = texts[2].replace(".", "_")
            base_model_path = os.path.abspath(os.path.join(
                Utils.get_models_path(), f"{host}_models"))

            file_name = texts[-1]
            file_name_no_ext = os.path.splitext(file_name)[0]
            file_ext = os.path.splitext(file_name)[1]
            md5_hash = Utils.Md5(url)

            save_path = os.path.join(
                base_model_path, f"{file_name_no_ext}.{md5_hash}{file_ext}")

            if os.path.exists(save_path) is False:
                if only_get_path:
                    return None
                os.makedirs(base_model_path, exist_ok=True)
                Utils.download_file(url, save_path)

            return save_path

    def print_log(*args):
        if os.environ.get("MZ_DEV", None) is not None:
            print(*args)

    def download_model(model_info, only_get_path=False):

        url = model_info["url"]
        output = model_info["output"]
        save_path = os.path.abspath(
            os.path.join(Utils.get_comfyui_models_path(), output))
        if not os.path.exists(save_path):
            if only_get_path:
                return None
            save_path = Utils.download_file(url, save_path)
        return save_path

    def load_lora(model, lora_path, strength_model):
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(
            model, None, lora, strength_model, 0)
        return model_lora

    def load_checkpoint(ckpt_name):
        cache_data = Utils.cache_get(ckpt_name)
        if cache_data is not None:
            print("load from cache: ", ckpt_name)
            return cache_data
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True)
        model, clip, vae = out[:3]
        Utils.cache_set(ckpt_name, (model, clip, vae))
        return model, clip, vae

    def file_hash(file_path, hash_method):
        if not os.path.isfile(file_path):
            return ''
        h = hash_method()
        with open(file_path, 'rb') as f:
            while b := f.read(8192):
                h.update(b)
        return h.hexdigest()

    def file_sha256(file_path):
        return Utils.file_hash(file_path, hashlib.sha256)

    def get_auto_model_fullpath(model_name):
        find_paths = []
        target_sha256 = ""
        file_path = ""
        download_url = ""
        for model in MODEL_ZOO:
            if model["model"] == model_name:
                find_paths = model["find_path"]
                target_sha256 = model["SHA256"]
                file_path = model["file_path"]
                download_url = model["url"]
                break

        if target_sha256 == "":
            raise ValueError(f"Model {model_name} not found in MODEL_ZOO")

        if os.path.exists(file_path):
            if Utils.file_sha256(file_path) != target_sha256:
                print(f"Model {model_name} file hash not match...")
            return file_path

        for find_path in find_paths:
            find_fullpath = os.path.join(
                folder_paths.get_output_directory(), find_path)

            if os.path.exists(find_fullpath):
                for root, dirs, files in os.walk(find_fullpath):
                    # 排除隐藏文件夹
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    for file in files:
                        if target_sha256 == Utils.file_sha256(os.path.join(root, file)):
                            return os.path.join(root, file)

        return Utils.download_model({"url": download_url, "output": file_path})

    def is_sd15_model(model):
        type_str = str(type(model.model.model_config).__name__)
        return "SD15" in type_str

    def is_sdxl_model(model):
        type_str = str(type(model.model.model_config).__name__)
        return "SDXL" in type_str

    def progress_bar(steps, taesd_type="sd1_5"):
        class pb:
            if taesd_type == "sd1_5":
                taesd_decoder_name = "taesd_decoder"
            elif taesd_type == "sdxl":
                taesd_decoder_name = "taesdxl_decoder"
            latent_rgb_factors = None
            latent_channels = 4

            def __init__(self, steps):
                self.steps = steps
                self.pbar = comfy.utils.ProgressBar(steps)

            def get_previewer(self):
                import latent_preview
                previewer = latent_preview.get_previewer(
                    'cuda', self)
                return previewer

            def update(self, step, total_steps, pil_img=None):
                try:
                    pil_img_info = ("JPEG", pil_img, 512)
                    if pil_img is None:
                        pil_img_info = None
                    if type(pil_img) == Tuple or type(pil_img) == list or type(pil_img) == tuple:
                        pil_img_info = pil_img
                    # print("pil_img_info:", type(pil_img), pil_img_info)
                    # print("step:", step, "total_steps:", total_steps)
                    self.pbar.update_absolute(
                        step, total_steps, pil_img_info)
                except Exception as e:
                    print("progress_bar:", e)
                    raise e

        return pb(steps)

    def get_free_port():
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('localhost', 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def Simple_Server(reader):
        import threading

        from http.server import HTTPServer, BaseHTTPRequestHandler

        port = Utils.get_free_port()
        is_running = {"value": True}

        def stop_server():
            is_running.update({"value": False})

        httpd = None

        class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(200)
                self.end_headers()
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
                if reader(data) == False:
                    stop_server()

            def log_message(self, format, *args):
                pass

        httpd = HTTPServer(('localhost', port), SimpleHTTPRequestHandler)

        def serve_forever():
            try:
                print(
                    "===========================httpd.serve_forever() start=======================================")
                httpd.serve_forever()
            except Exception as e:
                pass
            print(
                "===========================httpd.serve_forever() end=======================================")
        threading.Thread(target=serve_forever).start()

        def check_server_stop():
            while is_running["value"]:
                # print("is_running : ", is_running)
                time.sleep(1)
            httpd.shutdown()
            httpd.server_close()

        threading.Thread(target=check_server_stop).start()

        return stop_server, port

    def xy_image(pre_render_images, pre_render_texts_x, pre_render_texts_y):

        def get_common_prefix(pre_render_texts_x):
            if len(pre_render_texts_x) == 0:
                return ""
            common_prefix = ""
            for i in range(len(pre_render_texts_x[0])):
                c = pre_render_texts_x[0][i]
                for j in range(len(pre_render_texts_x)):
                    if pre_render_texts_x[j][i] != c:
                        return common_prefix
                else:
                    common_prefix += c

            return common_prefix

        common_prefix = get_common_prefix(pre_render_texts_x)

        if common_prefix != "":
            pre_render_texts_x = [
                x.replace(common_prefix, "") for x in pre_render_texts_x]

        x_enable_num = len(pre_render_images)
        y_enable_num = len(pre_render_images[0])
        max_width = 0
        max_height = 0
        for x in range(0, len(pre_render_images)):
            for y in range(0, len(pre_render_images[x])):
                if pre_render_images[x][y].width < 512:
                    org_width = pre_render_images[x][y].width
                    org_height = pre_render_images[x][y].height
                    pre_render_images[x][y] = pre_render_images[x][y].resize(
                        (512, int(org_height * 512 / org_width)))

                if pre_render_images[x][y].width > max_width:
                    max_width = pre_render_images[x][y].width
                if pre_render_images[x][y].height > max_height:
                    max_height = pre_render_images[x][y].height

        image_xy_canvas = Image.new(
            "RGB", (max_width * x_enable_num, max_height * y_enable_num))
        for i in range(x_enable_num):
            for j in range(y_enable_num):
                image_xy_canvas.paste(
                    pre_render_images[i][j], (max_width * i, max_height * j))

        # draw axis
        from PIL import ImageDraw, ImageFont
        import PIL

        pil_version = PIL.__version__
        if pil_version >= "10.0.0":
            def textsize(self, text, font):
                left, top, right, bottom = self.textbbox((0, 0), text, font)
                return right - left, bottom - top
            ImageDraw.ImageDraw.textsize = textsize

        full_padding = 260

        full_canvas_width = max_width * x_enable_num + full_padding * 2
        full_canvas_height = max_height * y_enable_num + full_padding * 2
        full_canvas = Image.new(
            "RGB", (full_canvas_width, full_canvas_height), "white")

        full_canvas.paste(image_xy_canvas, (full_padding, full_padding))

        draw = ImageDraw.Draw(full_canvas)

        # top
        draw.line((full_padding, full_padding, full_canvas_width - full_padding,
                  full_padding), fill="black", width=4)

        # left
        draw.line((full_padding, full_padding, full_padding,
                  full_canvas_height - full_padding), fill="black", width=4)

        # bottom
        draw.line((full_padding, full_canvas_height - full_padding, full_canvas_width - full_padding,
                   full_canvas_height - full_padding), fill="black", width=4)

        # right
        draw.line((full_canvas_width - full_padding, full_padding, full_canvas_width - full_padding,
                   full_canvas_height - full_padding), fill="black", width=4)

        font_fullpath = Utils.download_model(
            {
                "url": "https://www.modelscope.cn/api/v1/models/wailovet/MinusZoneAIModels/repo?Revision=master&FilePath=font%2FAlibabaPuHuiTi-2-75-SemiBold.ttf",
                "output": "font/AlibabaPuHuiTi-2-75-SemiBold.ttf",
            }
        )
        if os.path.exists(font_fullpath):
            font = ImageFont.truetype(font_fullpath, size=32,)
        else:
            font = ImageFont.load_default()
        for i in range(x_enable_num):
            textwidth, textheight = draw.textsize(
                pre_render_texts_x[i], font)
            offset_x = max_width * i + full_padding + \
                ((max_width - textwidth) // 2)
            offset_y = full_padding - textheight - 24

            draw.text((offset_x, offset_y),
                      pre_render_texts_x[i], font=font, fill="black")

        for j in range(y_enable_num):
            textwidth, textheight = draw.textsize(
                pre_render_texts_y[j], font)

            label_text = pre_render_texts_y[j]

            def is_exceed_width(t):
                textwidth, _ = draw.textsize(
                    t, font)
                return textwidth > max_width - 24

            if textwidth > full_padding - 24:
                # 超过宽度就换行
                label_text = ""
                for c in pre_render_texts_y[j]:
                    if is_exceed_width(label_text + c):
                        break
                    label_text += c

            offset_x = full_padding - textwidth - 24
            offset_y = max_height * j + full_padding + \
                ((max_height - textheight) // 2)

            draw.text((offset_x, offset_y),
                      label_text, font=font, fill="black")

        return full_canvas

    def get_models_by_folder(dir_path):
        models = []
        for root, dirs, files in os.walk(dir_path):

            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if file.endswith(".pth") or file.endswith(".pt") or file.endswith(".pkl") or file.endswith(".onnx") or file.endswith(".safetensors"):
                    models.append(os.path.join(root, file))
        return models

    def get_folders_by_folder(dir_path):
        folders = []
        for root, dirs, files in os.walk(dir_path):


            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for dir in dirs:
                folders.append(os.path.join(root, dir))
        return folders


from torch import nn


class CustomizeEmbedsModel(nn.Module):
    dtype = torch.float16
    shared = None
    # x = torch.zeros(1, 1, 256, 2048)
    x = None

    def __init__(self, *args, **kwargs):
        super().__init__()

    def to(self, *args, **kwargs):
        return self

    def forward(self, *args, **kwargs):
        # print("CustomizeEmbedsModel forward: args:", args)
        # print("CustomizeEmbedsModel forward: kwargs:", kwargs)
        input_ids = kwargs.get("input_ids", None)
        # if self.x is None:
        if True:
            if input_ids is None:
                batch_size = 1
            else:
                batch_size = input_ids.shape[0]

            attention_mask = kwargs.get("attention_mask")
            attention_mask_dim = attention_mask.shape[1]
            self.x = torch.zeros(1, batch_size, 256, 2048, dtype=self.dtype)

        if kwargs.get("output_hidden_states", False):
            return {
                "hidden_states": self.x.to("cuda"),
                "input_ids": torch.zeros(1, 1),
            }
        return self.x


class CustomizeTokenizer(dict):
    added_tokens_encoder = []
    input_ids = None
    attention_mask = None
    batch_size = 1

    def __init__(self, *args, **kwargs):
        self['added_tokens_encoder'] = self.added_tokens_encoder
        self['input_ids'] = self.input_ids
        self['attention_mask'] = self.attention_mask
        self.batch_size = kwargs.get("batch_size", 1)

    def tokenize(self, text):
        return text

    def __call__(self, *args, **kwargs):
        # print("CustomizeTokenizer args:", args)
        # print("CustomizeTokenizer kwargs:", kwargs)

        value = args[0]
        if isinstance(value, str):
            batch_size = 1
        else:
            batch_size = value.shape[0]

        # print(f"CustomizeTokenizer batch_size: {batch_size}")
        # if self.input_ids is not None:
        #     return self

        self.input_ids = torch.zeros(batch_size, 256)
        self.attention_mask = torch.zeros(batch_size, 256)
        self['input_ids'] = self.input_ids
        self['attention_mask'] = self.attention_mask

        # print("CustomizeTokenizer input_ids:", self.input_ids.shape)
        # print("CustomizeTokenizer attention_mask:", self.attention_mask.shape)

        return self


class CustomizeEmbeds():
    def __init__(self):
        super().__init__()
        self.tokenizer = CustomizeTokenizer()
        self.model = CustomizeEmbedsModel().to("cuda")
        self.max_length = 256


class CustomizeMT5Embedder(nn.Module):
    device = torch.device("cuda")

    def __init__(
        self,
        model_dir="t5-v1_1-xxl",
        model_kwargs=None,
        torch_dtype=None,
        use_tokenizer_only=False,
        max_length=128,
        batch_size=1,
    ):
        super().__init__()
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.max_length = max_length
        self.tokenizer = CustomizeTokenizer(
            batch_size=batch_size
        )
        self.model = CustomizeEmbedsModel().to("cuda")

    def gradient_checkpointing_enable(self):
        pass

    def gradient_checkpointing_disable(self):
        pass

    def get_tokens_and_mask(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        tokens = text_tokens_and_mask["input_ids"][0]
        mask = text_tokens_and_mask["attention_mask"][0]
        return tokens, mask

    def get_text_embeddings(self, texts, attention_mask=True, layer_index=-1):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        outputs = self.model(
            input_ids=text_tokens_and_mask["input_ids"],
            attention_mask=(
                text_tokens_and_mask["attention_mask"]
                if attention_mask
                else None
            ),
            output_hidden_states=True,
        )
        text_encoder_embs = outputs["hidden_states"][layer_index].detach()

        return text_encoder_embs, text_tokens_and_mask["attention_mask"].to(self.device)

    def get_input_ids(self, caption):
        return self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).input_ids

    def get_hidden_states(self, input_ids, layer_index=-1):
        return self.get_text_embeddings(input_ids, layer_index=layer_index)


class HSubprocess:
    process_instance = None
    process_instance_pid = None
    screen_name = None
    _mswindows = False

    def __init__(self, args, screen_name=None):
        self._mswindows = (sys.platform == "win32")
        if self._mswindows:
            self.args = ["start", "/w"]
            self.args.extend(args)
            return

        self.screen_name = screen_name
        if screen_name is not None:
            try:
                subprocess.check_call(["screen", "-v"])
            except Exception as e:
                raise Exception("Please install screen first.")

            screen_cmd = ["screen", "-R", screen_name, "-m", ]
            screen_cmd.extend(args)

            self.args = screen_cmd
        else:
            self.args = args

    def stop(self):
        if self._mswindows:
            print('taskkill /F /FI "WINDOWTITLE eq hook_kohya_ss_run" /T')
            os.system(
                f'taskkill /F /FI "WINDOWTITLE eq hook_kohya_ss_run" /T')
            

        if self.process_instance is not None:
            if self.screen_name is not None:
                subprocess.run(
                    ["screen", "-d", self.screen_name])

            self.process_instance.kill()
            self.process_instance = None

            try:
                try:
                    import psutil
                except ImportError:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "psutil"])
                    import psutil
                psutil.Process(self.process_instance_pid).terminate()
            except Exception as e:
                print(e)

            self.process_instance_pid = None

    def wait(self):
        with subprocess.Popen(
            self.args,
            stdin=subprocess.PIPE,
            shell=True,
        ) as process:
            self.process_instance = process
            self.process_instance_pid = process.pid
            print(f"Subprocess PID: {self.process_instance_pid}")
            try:
                stdout, stderr = process.communicate()
            except subprocess.TimeoutExpired as exc:
                process.kill()
                if self._mswindows:
                    exc.stdout, exc.stderr = process.communicate()
                else:
                    process.wait()
                raise
            except Exception as e:
                process.wait()
                raise
            retcode = process.poll()
            if retcode != 0:
                raise subprocess.CalledProcessError(retcode, process.args)

        self.process_instance = None
        self.process_instance_pid = None

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False