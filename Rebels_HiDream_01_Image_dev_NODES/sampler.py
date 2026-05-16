"""
Rebel HiDream-O1 Sampler.
Wraps upstream models.pipeline.generate_image().
Supports T2I, image editing with up to 4 refs,
full KSampler-style sampler/scheduler dropdowns, seam smoothing.
"""
import os
import tempfile
import numpy as np
import torch
from PIL import Image

try:
    import comfy.samplers
    SAMPLER_NAMES = list(comfy.samplers.KSampler.SAMPLERS)
    SCHEDULER_NAMES = list(comfy.samplers.KSampler.SCHEDULERS) + ["flash"]
except Exception:
    SAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde",
                     "dpmpp_sde", "uni_pc", "uni_pc_bh2", "ddim", "lcm", "deis"]
    SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple",
                       "ddim_uniform", "beta", "flash"]

STOCHASTIC_SAMPLERS = {
    "euler_ancestral", "euler_ancestral_cfg_pp",
    "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde", "dpmpp_sde_gpu",
    "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
    "ddpm",
}

UNIPC_SAMPLERS = {"uni_pc", "uni_pc_bh2", "deis"}

RESOLUTION_PRESETS = {
    "2048x2048 (1:1 square)":      (2048, 2048),
    "2304x1728 (4:3 landscape)":   (2304, 1728),
    "1728x2304 (3:4 portrait)":    (1728, 2304),
    "2560x1440 (16:9 landscape)":  (2560, 1440),
    "1440x2560 (9:16 portrait)":   (1440, 2560),
    "2496x1664 (3:2 landscape)":   (2496, 1664),
    "1664x2496 (2:3 portrait)":    (1664, 2496),
    "3104x1312 (21:9 ultrawide)":  (3104, 1312),
    "1312x3104 (9:21 tall)":       (1312, 3104),
    "2304x1792 (~9:7 landscape)":  (2304, 1792),
    "1792x2304 (~7:9 portrait)":   (1792, 2304),
    "Custom (force, may break)":   None,
}


def _image_to_path(image_tensor, temp_dir, index):
    img_np = (image_tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    path = os.path.join(temp_dir, f"ref_{index}.png")
    pil_img.save(path)
    return path


def _resolve_scheduler(sampler_name, scheduler_name):
    from models.pipeline import DEFAULT_TIMESTEPS
    if scheduler_name == "flash":
        return "flash", DEFAULT_TIMESTEPS
    if sampler_name in UNIPC_SAMPLERS:
        return "default", None
    if sampler_name in STOCHASTIC_SAMPLERS:
        return "flash", None
    return "flash", None


class RebelHiDreamO1Sampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":  ("HIDREAM_O1_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic photograph of a fox in a snowy forest at golden hour",
                }),
                "resolution_preset": (list(RESOLUTION_PRESETS.keys()),
                                      {"default": "2048x2048 (1:1 square)"}),
                "custom_width":  ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                          "tooltip": "Only used when resolution is Custom."}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                          "tooltip": "Only used when resolution is Custom."}),
                "sampler": (SAMPLER_NAMES, {"default": "euler_ancestral",
                    "tooltip": "Dev: euler_ancestral | Full: uni_pc"}),
                "scheduler": (SCHEDULER_NAMES, {"default": "flash",
                    "tooltip": "Dev: flash | Full: normal"}),
                "steps": ("INT",   {"default": 28,  "min": 1,   "max": 100,
                                    "tooltip": "Dev: 28 | Full: 50"}),
                "cfg":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1,
                                    "tooltip": "Dev: 0.0 | Full: 5.0"}),
                "shift": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1,
                                    "tooltip": "Dev: 1.0 | Full: 3.0"}),
                "seed":  ("INT",   {"default": 32,  "min": 0,   "max": 0xffffffffffffffff}),
            },
            "optional": {
                "ref_image_1": ("IMAGE", {"tooltip": "Reference image 1 for editing."}),
                "ref_image_2": ("IMAGE", {"tooltip": "Reference image 2."}),
                "ref_image_3": ("IMAGE", {"tooltip": "Reference image 3."}),
                "ref_image_4": ("IMAGE", {"tooltip": "Reference image 4."}),
                "keep_original_aspect": ("BOOLEAN", {"default": False,
                    "tooltip": "With 1 ref image, output matches that image's aspect ratio."}),
                "noise_scale_start": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "noise_scale_end":   ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "noise_clip_std":    ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seam_smooth_steps": ("INT",   {"default": 0, "min": 0, "max": 10, "step": 1,
                                                "tooltip": "Final steps with seam smoothing. 0 = off. Try 3-5 for Full."}),
                "seam_smooth_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                                   "tooltip": "Blend strength for smoothing."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Rebels/HiDream-O1"

    def sample(self, model, prompt, resolution_preset,
               custom_width, custom_height,
               sampler, scheduler, steps, cfg, shift, seed,
               ref_image_1=None, ref_image_2=None,
               ref_image_3=None, ref_image_4=None,
               keep_original_aspect=False,
               noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5,
               seam_smooth_steps=0, seam_smooth_strength=0.5):
        generate_image    = model["generate_image"]
        DEFAULT_TIMESTEPS = model["DEFAULT_TIMESTEPS"]

        resolved_scheduler, timesteps_list = _resolve_scheduler(sampler, scheduler)
        print(f"[Rebels_HiDream_O1] Sampler: {sampler} → pipeline scheduler: {resolved_scheduler}")

        res = RESOLUTION_PRESETS[resolution_preset]
        force_custom = res is None
        if force_custom:
            width, height = custom_width, custom_height
            print(f"[Rebels_HiDream_O1] Custom resolution forced: {width}x{height}")
        else:
            width, height = res

        ref_paths = []
        temp_dir = None
        connected = [r for r in [ref_image_1, ref_image_2, ref_image_3, ref_image_4] if r is not None]
        if connected:
            temp_dir = tempfile.mkdtemp(prefix="hidream_refs_")
            for i, ref in enumerate(connected):
                ref_paths.append(_image_to_path(ref, temp_dir, i))
            print(f"[Rebels_HiDream_O1] {len(ref_paths)} reference image(s) loaded")

        patched_originals = {}
        if force_custom:
            try:
                import models.pipeline as _pipeline
                if hasattr(_pipeline, "find_closest_resolution"):
                    patched_originals["pipeline"] = _pipeline.find_closest_resolution
                    _pipeline.find_closest_resolution = lambda w, h: (w, h)
                import models.utils as _utils
                if hasattr(_utils, "find_closest_resolution"):
                    patched_originals["utils"] = _utils.find_closest_resolution
                    _utils.find_closest_resolution = lambda w, h: (w, h)
            except Exception as e:
                print(f"[Rebels_HiDream_O1] Could not patch find_closest_resolution: {e}")

        try:
            pil_image = generate_image(
                model=model["model"],
                processor=model["processor"],
                prompt=prompt,
                ref_image_paths=ref_paths,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=cfg,
                shift=shift,
                timesteps_list=timesteps_list,
                scheduler_name=resolved_scheduler,
                seed=seed,
                noise_scale_start=noise_scale_start,
                noise_scale_end=noise_scale_end,
                noise_clip_std=noise_clip_std,
                keep_original_aspect=keep_original_aspect,
                seam_smooth_steps=seam_smooth_steps,
                seam_smooth_strength=seam_smooth_strength,
            )
        finally:
            if "pipeline" in patched_originals:
                import models.pipeline as _pipeline
                _pipeline.find_closest_resolution = patched_originals["pipeline"]
            if "utils" in patched_originals:
                import models.utils as _utils
                _utils.find_closest_resolution = patched_originals["utils"]
            if temp_dir:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

        arr = np.asarray(pil_image.convert("RGB")).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).unsqueeze(0)
        return (image_tensor,)
