<img width="1517" height="905" alt="Screenshot (116)" src="https://github.com/user-attachments/assets/84989f70-01ff-4329-951b-35a2f10f721f" />


# Rebels_HiDream-01_Image_Dev_NODES

# NOW SUPPORTS IMAGE DEV 2604 !!!

# NOW WITH NEW AND IMPROVED LORA SUPPORT: features a lora injection INSIDE the loader node itself to remove any issues with memory stalling due to lora stack injector node for low vram. this will load the entire lora without OOMing or missing pairs during merge due to lack of memory.

Now featuring full Sampler/Scheduler support, Multi-Reference Editing, and built-in Seam Smoothing to eliminate tiling artifacts.

This repository provides high-performance custom ComfyUI nodes for running the HiDream-01-Image-Dev models (both BF16 and GGUF). As a VAE-less, Pixel-Level Unified Transformer, HiDream-O1 generates raw pixels token-by-token. These nodes are optimized for local hardware, utilizing upfront dequantization and aggressive system RAM offloading—perfect for 8GB VRAM cards like the RTX 3070 by leveraging your 16GB of system RAM.

# 🚀 Key Features

- Multi-Reference Editing: Inject up to 4 reference images to guide your generations.

- Integrated LoRA Stack: Manage up to 4 LoRAs with fingerprint-based no-op detection.

- Advanced Seam Smoothing: Built-in "Patch Model Smoothing" logic to fix bad tiling and textures.

- Expanded Sampler Support: Full flexibility with native ComfyUI samplers and schedulers.

- Seam Visualizer: Heatmap-based analysis to monitor and perfect generation consistency.

# 📦 Prerequisites
Clone Upstream Repo:
git clone [https://github.com/HiDream-ai/HiDream-O1-Image.git](https://github.com/HiDream-ai/HiDream-O1-Image.git)

# Download weights:

BF16: Place in ComfyUI/models/checkpoints/

GGUF: Place in ComfyUI/models/diffusion_models/

# 🛠️ Installation
[!IMPORTANT]
DO NOT install or run these nodes from a OneDrive-synced folder. Ensure your ComfyUI installation is on a strict local system drive (C: or D:) to avoid pathing and virtual environment errors.

Navigate to ComfyUI/custom_nodes.

git clone [https://github.com/RealRebelAI/Rebels_HiDream-01_Image_Dev_NODES.git](https://github.com/RealRebelAI/Rebels_HiDream-01_Image_Dev_NODES.git)

Install requirements:

Portable: ../../../python_embeded/python.exe -m pip install -r requirements.txt

Standard: pip install -r requirements.txt

# 🧩 Node Documentation
1. Rebel HiDream-O1 Loaders (GGUF & BF16)
offload:

aggressive: Recommended for 8GB VRAM cards. Moves model weights to system RAM.

balanced: Splits weights between VRAM and system RAM.

Note: The GGUF loader will pause at 100% while unpacking bytes into PyTorch tensors. This is expected and prevents CPU bottlenecks during generation.

2. Rebel HiDream-O1 LoRA Stack Injector
Inject multiple LoRAs into the model stream.

Slots: 4 LoRA slots with independent strength and bypass toggles.

Efficiency: Uses fingerprint-based detection to ensure no-op slots don't impact compute time.

Compatibility: If using aggressive offloading on BF16 paths, use the Seam Visualizer to confirm LoRA effects, as accelerate may occasionally reset in-place merges.

3. Rebel HiDream-O1 Sampler
The core engine of the suite, now heavily updated for precision and texture control.

Multi-Ref Inputs: ref_image_1 through ref_image_4.

Resolution Preset: 2048x2048 is native. Lower resolutions (1024x1024) will automatically "snap" to the closest supported token sequence length.

Sampler/Scheduler: Supports all native options.


CFG: For the Dev model, 0.0 is the recipe default. For higher texture/guidance with the detail scheduler, 5.0 is a strong starting point.

Seam Smoothing Suite:

seam_smooth_steps: Number of steps to apply patch smoothing.

seam_smooth_strength: Intensity of the tiling correction.

seam_adaptive_threshold: Dynamically targets inconsistent patches.

4. Rebel HiDream-O1 Seam Visualizer
Analyze the "health" of your generation's tiling.

Heatmap: Generates an overlay showing which patches are struggling with consistency.

Colormap: Inferno, Magma, or Viridis for clear contrast.

Usage: Connect to the IMAGE output of the sampler to dial in your seam_smooth settings.

# ⚠️ Known Behaviors
Compute Times: Because this is a pixel-transformer rendering 2048x2048 images token-by-token without a VAE, expect significantly longer generation times compared to latent models like Flux or SDXL.

Resolution Snapping: Entering custom resolutions (e.g., 512x512) will result in the model snapping to its pre-trained position embeddings (typically 2048x2048).

Prompting: Avoid over-stacking "8k, ultra-detailed" tags unless you want a heavily illustrated look. The model is naturally responsive to clean, photographic descriptions.
