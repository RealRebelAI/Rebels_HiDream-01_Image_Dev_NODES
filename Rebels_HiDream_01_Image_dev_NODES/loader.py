"""
Rebel HiDream-O1 Loaders.
  - RebelHiDreamO1Loader      → GGUF path
  - RebelHiDreamO1LoaderHF    → bf16 single-file safetensors
Both return a HIDREAM_O1_MODEL handle the sampler can drive.

Upstream models/ (pipeline.py, qwen3_vl_transformers.py, etc.) are vendored
directly inside this node pack — no separate clone needed.
"""
import os
import sys
import torch
import folder_paths
from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map

from .gguf_ops import load_gguf


# ---------------------------------------------------------------------------
# Auto-detect vendored models/ folder and add to sys.path
# ---------------------------------------------------------------------------
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDORED_PIPELINE = os.path.join(_NODE_DIR, "models", "pipeline.py")
if os.path.isfile(_VENDORED_PIPELINE):
    if _NODE_DIR not in sys.path:
        sys.path.insert(0, _NODE_DIR)
else:
    raise FileNotFoundError(
        f"Vendored models/pipeline.py not found at {_VENDORED_PIPELINE}\n"
        f"Copy the upstream HiDream-O1-Image/models/ folder into:\n"
        f"  {os.path.join(_NODE_DIR, 'models')}"
    )


# Reuse ComfyUI's already-resolved diffusion_models paths so that any locations
# declared in extra_model_paths.yaml (e.g. NVME drives) are honored, instead of
# hardcoding only <models_dir>/diffusion_models.
_DIFFUSION_MODELS_DIRS = folder_paths.get_folder_paths("diffusion_models")
if "hidream_o1" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["hidream_o1"] = (
        list(_DIFFUSION_MODELS_DIRS),
        {".gguf", ".safetensors"},
    )


_SPECIAL_TOKENS = {
    "boi_token": "<|boi_token|>",
    "bor_token": "<|bor_token|>",
    "eor_token": "<|eor_token|>",
    "bot_token": "<|bot_token|>",
    "tms_token": "<|tms_token|>",
}


OFFLOAD_PRESETS = {
    "aggressive": {"cuda_gb": 5.5,  "cpu_gb": 13.0},
    "balanced":   {"cuda_gb": 9.5,  "cpu_gb": 20.0},
    "minimal":    {"cuda_gb": 22.0, "cpu_gb": 8.0},
}


def _add_special_tokens(tokenizer):
    for attr, tok in _SPECIAL_TOKENS.items():
        setattr(tokenizer, attr, tok)


# ===========================================================================
# GGUF loader
# ===========================================================================
class RebelHiDreamO1Loader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_name": (folder_paths.get_filename_list("hidream_o1"),),
                "config_path": ("STRING", {
                    "default": "HiDream-ai/HiDream-O1-Image-Dev",
                    "multiline": False,
                    "tooltip": "HF repo id OR local folder for config.json + tokenizer files.",
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "offload": (list(OFFLOAD_PRESETS.keys()), {"default": "aggressive"}),
                "compute_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("HIDREAM_O1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Rebels/HiDream-O1"

    def load(self, gguf_name, config_path, device, offload, compute_dtype):
        gguf_path = folder_paths.get_full_path("hidream_o1", gguf_name)
        if gguf_path is None or not os.path.isfile(gguf_path):
            raise FileNotFoundError(f"GGUF '{gguf_name}' not found in {_DIFFUSION_MODELS_DIRS}")

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }[compute_dtype]
        preset = OFFLOAD_PRESETS[offload]

        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        from models.pipeline import generate_image, DEFAULT_TIMESTEPS

        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration(config)

        print(f"[Rebels_HiDream_O1] Loading GGUF: {gguf_path}")
        missing, unexpected = load_gguf(gguf_path, model, target_dtype=torch_dtype)
        if missing:
            print(f"[Rebels_HiDream_O1] WARN: {len(missing)} missing keys: {list(missing)[:5]}")
        if unexpected:
            print(f"[Rebels_HiDream_O1] WARN: {len(unexpected)} unexpected keys: {list(unexpected)[:5]}")

        if device == "cuda" and torch.cuda.is_available() and offload != "minimal":
            max_memory = {
                0:     f"{preset['cuda_gb']:.1f}GiB",
                "cpu": f"{preset['cpu_gb']:.1f}GiB",
            }
            device_map = infer_auto_device_map(
                model, max_memory=max_memory, dtype=torch_dtype,
                no_split_module_classes=["Qwen3VLDecoderLayer"],
            )
            model = dispatch_model(model, device_map=device_map)
        elif device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        model.eval()

        processor = AutoProcessor.from_pretrained(config_path)
        tokenizer = (
            processor
            if isinstance(processor, PreTrainedTokenizerBase)
            else processor.tokenizer
        )
        _add_special_tokens(tokenizer)

        return ({
            "model":             model,
            "processor":         processor,
            "tokenizer":         tokenizer,
            "generate_image":    generate_image,
            "DEFAULT_TIMESTEPS": DEFAULT_TIMESTEPS,
            "device":            device,
            "dtype":             torch_dtype,
            "offload":           offload,
        },)


# ===========================================================================
# BF16 / single-file safetensors loader
# ===========================================================================
class RebelHiDreamO1LoaderHF:

    @classmethod
    def INPUT_TYPES(cls):
        try:
            ckpts = [f for f in folder_paths.get_filename_list("checkpoints")
                     if f.lower().endswith(".safetensors")]
        except Exception:
            ckpts = []
        return {
            "required": {
                "safetensors_file": (ckpts if ckpts else ["<no .safetensors in checkpoints/>"],),
                "config_path": ("STRING", {
                    "default": "HiDream-ai/HiDream-O1-Image-Dev",
                    "multiline": False,
                    "tooltip": "HF repo id OR local folder for config.json + tokenizer files.",
                }),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "offload": (list(OFFLOAD_PRESETS.keys()), {"default": "aggressive"}),
                "offload_folder": ("STRING", {
                    "default": "hidream_offload",
                    "tooltip": "Disk offload folder for layers that don't fit in VRAM+RAM.",
                }),
            }
        }

    RETURN_TYPES = ("HIDREAM_O1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Rebels/HiDream-O1"

    def load(self, safetensors_file, config_path, dtype, offload, offload_folder):
        from accelerate import load_checkpoint_and_dispatch

        sft_path = folder_paths.get_full_path("checkpoints", safetensors_file)
        if sft_path is None or not os.path.isfile(sft_path):
            raise FileNotFoundError(
                f"safetensors file not found: {safetensors_file}\n"
                f"Expected in ComfyUI/models/checkpoints/"
            )

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        preset = OFFLOAD_PRESETS[offload]

        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        from models.pipeline import generate_image, DEFAULT_TIMESTEPS

        config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration(config)

        os.makedirs(offload_folder, exist_ok=True)
        max_memory = {
            0:     f"{preset['cuda_gb']:.1f}GiB",
            "cpu": f"{preset['cpu_gb']:.1f}GiB",
        }

        print(f"[Rebels_HiDream_O1] Loading bf16 safetensors: {sft_path}")
        print(f"[Rebels_HiDream_O1] Memory budget: {max_memory}, disk offload: {offload_folder}")

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=sft_path,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder,
            no_split_module_classes=["Qwen3VLDecoderLayer"],
            dtype=torch_dtype,
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(config_path, trust_remote_code=True)
        tokenizer = (
            processor
            if isinstance(processor, PreTrainedTokenizerBase)
            else processor.tokenizer
        )
        _add_special_tokens(tokenizer)

        return ({
            "model":             model,
            "processor":         processor,
            "tokenizer":         tokenizer,
            "generate_image":    generate_image,
            "DEFAULT_TIMESTEPS": DEFAULT_TIMESTEPS,
            "device":            "cuda",
            "dtype":             torch_dtype,
            "offload":           offload,
        },)
