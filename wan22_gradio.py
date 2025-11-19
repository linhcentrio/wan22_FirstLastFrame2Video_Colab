#!/usr/bin/env python3
"""
Wan2.2 First & Last Frame to Video Generator
Refactored from Jupyter Notebook to standalone Python script
Supports both CLI and Gradio UI modes
"""

import os
import sys
import gc
import random
import shutil
import subprocess
import glob
from pathlib import Path
from typing import Optional, Tuple, List
import argparse

import torch
import numpy as np
import cv2
from PIL import Image
import imageio

# ComfyUI dependencies
sys.path.insert(0, '/content/ComfyUI')

from comfy import model_management
from nodes import (
    CLIPLoader, CLIPTextEncode, VAEDecode, VAELoader,
    KSamplerAdvanced, LoadImage, CLIPVisionLoader,
    CLIPVisionEncode, LoraLoaderModelOnly, ImageScale
)
from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
from custom_nodes.ComfyUI_KJNodes.nodes.model_optimization_nodes import (
    WanVideoTeaCacheKJ, PathchSageAttentionKJ, WanVideoNAG
)
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_wan import WanFirstLastFrameToVideo


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

class Config:
    """Configuration constants for the application"""
    COMFYUI_ROOT = "/content/ComfyUI"
    RIFE_ROOT = "/content/Practical-RIFE"
    
    # Model paths
    MODELS_DIR = f"{COMFYUI_ROOT}/models"
    DIFFUSION_MODELS = f"{MODELS_DIR}/diffusion_models"
    LORAS = f"{MODELS_DIR}/loras"
    TEXT_ENCODERS = f"{MODELS_DIR}/text_encoders"
    VAE = f"{MODELS_DIR}/vae"
    CLIP_VISION = f"{MODELS_DIR}/clip_vision"
    
    # Input/Output
    INPUT_DIR = f"{COMFYUI_ROOT}/input"
    OUTPUT_DIR = f"{COMFYUI_ROOT}/output"
    
    # Default model URLs
    HIGH_NOISE_MODEL = "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q6_K.gguf"
    LOW_NOISE_MODEL = "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q6_K.gguf"
    HIGH_NOISE_LORA = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors"
    LOW_NOISE_LORA = "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_memory():
    """Clear GPU and system memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def download_with_aria2c(url: str, dest_dir: str, filename: str = None, silent: bool = True) -> str:
    """
    Download file using aria2c with multi-threading
    
    Args:
        url: Download URL
        dest_dir: Destination directory
        filename: Optional output filename
        silent: Suppress output
        
    Returns:
        Downloaded filename
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = url.split('/')[-1].split('?')[0]
    
    cmd = [
        'aria2c', '--console-log-level=error',
        '-c', '-x', '16', '-s', '16', '-k', '1M',
        '-d', dest_dir, '-o', filename, url
    ]
    
    if silent:
        cmd.extend(['--summary-interval=0', '--quiet'])
        print(f"Downloading {filename}...", end=' ', flush=True)
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if silent:
            print("Done!")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading {filename}: {e.stderr.strip()}")
        raise


def download_civitai_model(civitai_link: str, civitai_token: str, folder: str) -> str:
    """Download model from CivitAI"""
    os.makedirs(folder, exist_ok=True)
    
    try:
        model_id = civitai_link.split("/models/")[1].split("?")[0]
    except IndexError:
        raise ValueError("Invalid Civitai URL format")
    
    civitai_url = f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor"
    if civitai_token:
        civitai_url += f"&token={civitai_token}"
    
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"model_{timestamp}.safetensors"
    full_path = os.path.join(folder, filename)
    
    download_command = f'wget --max-redirect=10 --show-progress "{civitai_url}" -O "{full_path}"'
    print("Downloading from Civitai...")
    os.system(download_command)
    
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        print(f"✅ LoRA downloaded successfully: {full_path}")
    else:
        print(f"❌ LoRA download failed: {full_path}")
    
    return filename


def image_width_height(image: torch.Tensor) -> Tuple[int, int]:
    """Extract width and height from image tensor"""
    if image.ndim == 4:
        _, height, width, _ = image.shape
    elif image.ndim == 3:
        height, width, _ = image.shape
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return width, height


def save_as_mp4(images: List[torch.Tensor], filename_prefix: str, fps: int, 
                output_dir: str = None) -> str:
    """Save image sequence as MP4 video"""
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.mp4"
    
    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]
    
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    
    return output_path


def save_as_image(image: torch.Tensor, filename_prefix: str, 
                  output_dir: str = None) -> str:
    """Save single frame as PNG"""
    if output_dir is None:
        output_dir = Config.OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.png"
    
    frame = (image.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(frame).save(output_path)
    
    return output_path


# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

class EnvironmentSetup:
    """Handle environment setup and model downloads"""
    
    @staticmethod
    def install_dependencies():
        """Install required packages"""
        print("📦 Installing dependencies...")
        
        commands = [
            "pip install -q torch==2.6.0 torchvision==0.21.0",
            "pip install -q torchsde einops diffusers accelerate xformers==0.0.29.post2 triton==3.2.0 sageattention==1.0.6",
            "pip install -q av spandrel albumentations insightface onnx opencv-python segment_anything ultralytics onnxruntime onnxruntime-gpu",
            "apt -y install -qq aria2 ffmpeg"
        ]
        
        for cmd in commands:
            os.system(cmd + " > /dev/null 2>&1")
        
        print("✅ Dependencies installed")
    
    @staticmethod
    def clone_repositories():
        """Clone required repositories"""
        print("📥 Cloning repositories...")
        
        # ComfyUI
        if not os.path.exists(Config.COMFYUI_ROOT):
            os.system(f"git clone --branch ComfyUI_v0.3.47 https://github.com/Isi-dev/ComfyUI {Config.COMFYUI_ROOT} > /dev/null 2>&1")
        
        # Custom nodes
        custom_nodes_dir = f"{Config.COMFYUI_ROOT}/custom_nodes"
        os.makedirs(custom_nodes_dir, exist_ok=True)
        
        if not os.path.exists(f"{custom_nodes_dir}/ComfyUI_GGUF"):
            os.system(f"cd {custom_nodes_dir} && git clone --branch forHidream https://github.com/Isi-dev/ComfyUI_GGUF.git > /dev/null 2>&1")
            os.system(f"cd {custom_nodes_dir}/ComfyUI_GGUF && pip install -r requirements.txt > /dev/null 2>&1")
        
        if not os.path.exists(f"{custom_nodes_dir}/ComfyUI_KJNodes"):
            os.system(f"cd {custom_nodes_dir} && git clone --branch kjnv1.1.3 https://github.com/Isi-dev/ComfyUI_KJNodes.git > /dev/null 2>&1")
            os.system(f"cd {custom_nodes_dir}/ComfyUI_KJNodes && pip install -r requirements.txt > /dev/null 2>&1")
        
        # Practical-RIFE
        if not os.path.exists(Config.RIFE_ROOT):
            os.system(f"git clone https://github.com/Isi-dev/Practical-RIFE {Config.RIFE_ROOT} > /dev/null 2>&1")
            os.system(f"pip install git+https://github.com/rk-exxec/scikit-video.git@numpy_deprecation > /dev/null 2>&1")
            
            # Download RIFE models
            train_log = f"{Config.RIFE_ROOT}/train_log"
            os.makedirs(train_log, exist_ok=True)
            
            rife_files = [
                "IFNet_HDv3.py", "RIFE_HDv3.py", "refine.py", "flownet.pkl"
            ]
            for file in rife_files:
                url = f"https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/{file}"
                os.system(f"wget -q {url} -O {train_log}/{file}")
        
        print("✅ Repositories cloned")
    
    @staticmethod
    def download_base_models():
        """Download essential models"""
        print("📥 Downloading base models...")
        
        # Text encoder
        download_with_aria2c(
            "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            Config.TEXT_ENCODERS
        )
        
        # VAE
        download_with_aria2c(
            "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/wan_2.1_vae.safetensors",
            Config.VAE
        )
        
        # CLIP Vision
        download_with_aria2c(
            "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/clip_vision_h.safetensors",
            Config.CLIP_VISION
        )
        
        print("✅ Base models downloaded")
    
    @staticmethod
    def download_wan_models(high_noise_url: str = None, low_noise_url: str = None):
        """Download Wan diffusion models"""
        print("📥 Downloading Wan models...")
        
        if high_noise_url is None:
            high_noise_url = Config.HIGH_NOISE_MODEL
        if low_noise_url is None:
            low_noise_url = Config.LOW_NOISE_MODEL
        
        high_model = download_with_aria2c(high_noise_url, Config.DIFFUSION_MODELS)
        low_model = download_with_aria2c(low_noise_url, Config.DIFFUSION_MODELS)
        
        print("✅ Wan models downloaded")
        return high_model, low_model
    
    @staticmethod
    def download_speed_loras(high_lora_url: str = None, low_lora_url: str = None):
        """Download speed optimization LoRAs"""
        print("📥 Downloading speed LoRAs...")
        
        if high_lora_url is None:
            high_lora_url = Config.HIGH_NOISE_LORA
        if low_lora_url is None:
            low_lora_url = Config.LOW_NOISE_LORA
        
        high_lora = download_with_aria2c(high_lora_url, Config.LORAS)
        low_lora = download_with_aria2c(low_lora_url, Config.LORAS)
        
        print("✅ Speed LoRAs downloaded")
        return high_lora, low_lora
    
    @classmethod
    def setup_all(cls):
        """Run complete setup"""
        print("🚀 Starting environment setup...")
        
        # Set CUDA memory config
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        cls.install_dependencies()
        cls.clone_repositories()
        cls.download_base_models()
        high_model, low_model = cls.download_wan_models()
        high_lora, low_lora = cls.download_speed_loras()
        
        print("✅ Environment setup complete!")
        
        return {
            'high_noise_model': high_model,
            'low_noise_model': low_model,
            'high_noise_lora': high_lora,
            'low_noise_lora': low_lora
        }


# ============================================================================
# VIDEO GENERATOR
# ============================================================================

class VideoGenerator:
    """Main video generation class"""
    
    def __init__(self, models: dict):
        """
        Initialize generator with model paths
        
        Args:
            models: Dictionary with model filenames
        """
        self.models = models
        self.output_path = ""
        
        # Initialize nodes
        self.unet_loader = UnetLoaderGGUF()
        self.sage_attention = PathchSageAttentionKJ()
        self.wan_nag = WanVideoNAG()
        self.teacache = WanVideoTeaCacheKJ()
        self.model_sampling = ModelSamplingSD3()
        self.clip_loader = CLIPLoader()
        self.clip_encode = CLIPTextEncode()
        self.vae_loader = VAELoader()
        self.clip_vision_loader = CLIPVisionLoader()
        self.clip_vision_encode = CLIPVisionEncode()
        self.load_image = LoadImage()
        self.wan_f2v = WanFirstLastFrameToVideo()
        self.ksampler = KSamplerAdvanced()
        self.vae_decode = VAEDecode()
        self.lora_loader = LoraLoaderModelOnly()
        self.image_scaler = ImageScale()
    
    def generate(
        self,
        image_path: str,
        image_path2: Optional[str] = None,
        positive_prompt: str = "a cute anime girl walking",
        negative_prompt: str = "low quality, blurry, static",
        width: int = 832,
        height: int = 480,
        frames: int = 81,
        fps: int = 16,
        steps: int = 20,
        high_noise_steps: int = 10,
        cfg_scale: float = 1.0,
        seed: int = None,
        sampler_name: str = "uni_pc",
        scheduler: str = "simple",
        use_sage_attention: bool = True,
        use_flow_shift: bool = True,
        flow_shift: float = 8.0,
        flow_shift2: float = 8.0,
        teacache_thresh: float = 0.275,
        use_high_speed_lora: bool = False,
        high_speed_strength: float = 0.8,
        use_low_speed_lora: bool = False,
        low_speed_strength: float = 1.2,
        output_format: str = "mp4",
        **kwargs
    ) -> str:
        """
        Generate video from first (and optionally last) frame
        
        Args:
            image_path: Path to first frame
            image_path2: Optional path to last frame
            positive_prompt: Text description of desired motion
            negative_prompt: Text description of unwanted content
            width: Output width
            height: Output height
            frames: Number of frames to generate
            fps: Frames per second
            steps: Total sampling steps
            high_noise_steps: Steps for high noise model
            cfg_scale: CFG guidance scale
            seed: Random seed (None for random)
            sampler_name: Sampler algorithm
            scheduler: Scheduler type
            use_sage_attention: Enable SageAttention optimization
            use_flow_shift: Enable flow shift
            flow_shift: High noise flow shift value
            flow_shift2: Low noise flow shift value
            teacache_thresh: TeaCache threshold
            use_high_speed_lora: Use high noise speed LoRA
            high_speed_strength: High speed LoRA strength
            use_low_speed_lora: Use low noise speed LoRA
            low_speed_strength: Low speed LoRA strength
            output_format: Output format (mp4, webm, png)
            
        Returns:
            Path to generated video/image
        """
        
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        
        print(f"🎬 Generating video with seed: {seed}")
        
        with torch.inference_mode():
            # Load text encoder
            print("Loading text encoder...")
            clip = self.clip_loader.load_clip(
                "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default"
            )[0]
            
            positive = self.clip_encode.encode(clip, positive_prompt)[0]
            negative = self.clip_encode.encode(clip, negative_prompt)[0]
            
            del clip
            clear_memory()
            
            # Load CLIP Vision
            clip_vision = self.clip_vision_loader.load_clip("clip_vision_h.safetensors")[0]
            
            # Load and process first image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            loaded_image = self.load_image.load_image(image_path)[0]
            width_orig, height_orig = image_width_height(loaded_image)
            
            if height == 0:
                height = int(width * height_orig / width_orig)
            
            print(f"Scaling image from {width_orig}x{height_orig} to {width}x{height}")
            loaded_image = self.image_scaler.upscale(
                loaded_image, "lanczos", width, height, "disabled"
            )[0]
            
            clip_vision_output = self.clip_vision_encode.encode(
                clip_vision, loaded_image, "none"
            )[0]
            
            # Load second image if provided
            if image_path2 is not None and os.path.exists(image_path2):
                loaded_image2 = self.load_image.load_image(image_path2)[0]
                loaded_image2 = self.image_scaler.upscale(
                    loaded_image2, "lanczos", width, height, "disabled"
                )[0]
                clip_vision_output2 = self.clip_vision_encode.encode(
                    clip_vision, loaded_image2, "none"
                )[0]
            else:
                loaded_image2 = None
                clip_vision_output2 = None
            
            del clip_vision
            clear_memory()
            
            # Load VAE and encode
            print("Loading VAE...")
            vae = self.vae_loader.load_vae("wan_2.1_vae.safetensors")[0]
            
            positive_out, negative_out, latent = self.wan_f2v.encode(
                positive, negative, vae, width, height, frames, 1,
                loaded_image, loaded_image2,
                clip_vision_output, clip_vision_output2
            )
            
            # ===== HIGH NOISE PASS =====
            print("Loading high noise model...")
            model = self.unet_loader.load_unet(self.models['high_noise_model'])[0]
            model = self.wan_nag.patch(model, negative, 11.0, 0.25, 2.5)[0]
            
            if use_flow_shift:
                model = self.model_sampling.patch(model, flow_shift)[0]
            
            used_steps = steps
            if use_high_speed_lora:
                print("Loading high noise speed LoRA...")
                model = self.lora_loader.load_lora_model_only(
                    model, self.models['high_noise_lora'], high_speed_strength
                )[0]
                used_steps = 4
            
            if use_sage_attention:
                model = self.sage_attention.patch(model, "auto")[0]
            
            if teacache_thresh > 0:
                print("Enabling TeaCache...")
                model = self.teacache.patch_teacache(
                    model, teacache_thresh, 0.1, 1.0, "main_device", "14B"
                )[0]
            
            print("Sampling high noise...")
            sampled = self.ksampler.sample(
                model=model,
                add_noise="enable",
                noise_seed=seed,
                steps=used_steps,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive_out,
                negative=negative_out,
                latent_image=latent,
                start_at_step=0,
                end_at_step=high_noise_steps,
                return_with_leftover_noise="enable"
            )[0]
            
            del model
            clear_memory()
            
            # ===== LOW NOISE PASS =====
            print("Loading low noise model...")
            model = self.unet_loader.load_unet(self.models['low_noise_model'])[0]
            
            if use_flow_shift:
                model = self.model_sampling.patch(model, flow_shift2)[0]
            
            used_steps = steps
            if use_low_speed_lora:
                print("Loading low noise speed LoRA...")
                model = self.lora_loader.load_lora_model_only(
                    model, self.models['low_noise_lora'], low_speed_strength
                )[0]
                used_steps = 4
            
            if use_sage_attention:
                model = self.sage_attention.patch(model, "auto")[0]
            
            if teacache_thresh > 0:
                model = self.teacache.patch_teacache(
                    model, teacache_thresh, 0.1, 1.0, "main_device", "14B"
                )[0]
            
            print("Sampling low noise...")
            sampled = self.ksampler.sample(
                model=model,
                add_noise="disable",
                noise_seed=seed,
                steps=used_steps,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive_out,
                negative=negative_out,
                latent_image=sampled,
                start_at_step=high_noise_steps,
                end_at_step=10000,
                return_with_leftover_noise="disable"
            )[0]
            
            del model
            clear_memory()
            
            # Decode
            print("Decoding latents...")
            decoded = self.vae_decode.decode(vae, sampled)[0]
            
            del vae
            clear_memory()
            
            # Save output
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"wan2_{timestamp}"
            
            if frames == 1:
                print("Saving as image...")
                self.output_path = save_as_image(decoded[0], base_name)
            else:
                if output_format.lower() == "mp4":
                    print("Saving as MP4...")
                    self.output_path = save_as_mp4(decoded, base_name, fps)
                else:
                    raise ValueError(f"Unsupported format: {output_format}")
            
            print(f"✅ Output saved: {self.output_path}")
            return self.output_path


# ============================================================================
# RIFE INTERPOLATION
# ============================================================================

def interpolate_with_rife(
    video_path: str,
    multiplier: int = 2,
    crf: int = 17,
    output_fps: int = 30
) -> str:
    """
    Interpolate video frames using RIFE
    
    Args:
        video_path: Input video path
        multiplier: Frame multiplication factor (2, 4, 8)
        crf: Output quality (lower = better)
        output_fps: Output FPS
        
    Returns:
        Path to interpolated video
    """
    print(f"🔄 Interpolating frames (x{multiplier})...")
    
    cwd_backup = os.getcwd()
    
    try:
        os.chdir(Config.RIFE_ROOT)
        
        cmd = [
            "python3", "inference_video.py",
            f"--multi={multiplier}",
            f"--fps={output_fps}",
            f"--video={video_path}",
            "--scale=1"
        ]
        
        env = os.environ.copy()
        env["XDG_RUNTIME_DIR"] = "/tmp"
        env["SDL_AUDIODRIVER"] = "dummy"
        env["FFMPEG_LOGLEVEL"] = "quiet"
        
        subprocess.run(cmd, env=env, check=True)
        
        # Find output
        video_files = glob.glob(f"{Config.OUTPUT_DIR}/*.mp4")
        latest = max(video_files, key=os.path.getctime)
        
        # Re-encode
        final_path = f"{Config.OUTPUT_DIR}/rife_{random.randint(0, 9999)}.mp4"
        subprocess.run([
            "ffmpeg", "-i", latest,
            "-vcodec", "libx264", "-crf", str(crf),
            "-preset", "fast", final_path,
            "-loglevel", "error", "-y"
        ], check=True)
        
        print(f"✅ Interpolation complete: {final_path}")
        return final_path
        
    finally:
        os.chdir(cwd_backup)


# ============================================================================
# GRADIO UI
# ============================================================================

def create_gradio_interface(generator: VideoGenerator):
    """Create Gradio web interface"""
    try:
        import gradio as gr
    except ImportError:
        os.system("pip install -q gradio")
        import gradio as gr
    
    def run_generation(
        img1, img2, prompt, neg_prompt,
        width, height, frames, fps,
        steps, high_steps, cfg,
        seed, use_sage, use_flow,
        shift1, shift2, teacache,
        use_high_lora, high_str,
        use_low_lora, low_str,
        do_rife, rife_mult, rife_crf
    ):
        try:
            if img1 is None:
                return None, "❌ First image required!"
            
            # Save images
            os.makedirs(Config.INPUT_DIR, exist_ok=True)
            img1_path = f"{Config.INPUT_DIR}/gradio_img1.png"
            img1.save(img1_path)
            
            img2_path = None
            if img2 is not None:
                img2_path = f"{Config.INPUT_DIR}/gradio_img2.png"
                img2.save(img2_path)
            
            if seed == 0:
                seed = random.randint(0, 2**32 - 1)
            
            # Generate
            output = generator.generate(
                image_path=img1_path,
                image_path2=img2_path,
                positive_prompt=prompt,
                negative_prompt=neg_prompt,
                width=int(width),
                height=int(height),
                frames=int(frames),
                fps=int(fps),
                steps=int(steps),
                high_noise_steps=int(high_steps),
                cfg_scale=cfg,
                seed=int(seed),
                use_sage_attention=use_sage,
                use_flow_shift=use_flow,
                flow_shift=shift1,
                flow_shift2=shift2,
                teacache_thresh=teacache,
                use_high_speed_lora=use_high_lora,
                high_speed_strength=high_str,
                use_low_speed_lora=use_low_lora,
                low_speed_strength=low_str
            )
            
            final_output = output
            status = f"✅ Generated! Seed: {seed}"
            
            # RIFE interpolation
            if do_rife and frames > 1:
                final_output = interpolate_with_rife(
                    output, int(rife_mult), int(rife_crf)
                )
                status += " | Interpolated"
            
            return final_output, status
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    # Build UI
    with gr.Blocks(title="Wan2.2 Video Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 Wan2.2 First & Last Frame to Video")
        
        with gr.Row():
            with gr.Column(scale=45):
                with gr.Group():
                    gr.Markdown("### Input Images")
                    with gr.Row():
                        img1 = gr.Image(label="First Frame (Required)", type="pil", height=300)
                        img2 = gr.Image(label="Last Frame (Optional)", type="pil", height=300)
                    
                    prompt = gr.Textbox(
                        label="Positive Prompt",
                        placeholder="Describe the motion...",
                        lines=3,
                        value="a cute anime girl walking"
                    )
                    
                    neg_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value="low quality, blurry, static",
                        lines=2
                    )
                
                with gr.Accordion("Settings", open=True):
                    with gr.Row():
                        width = gr.Slider(256, 1280, 832, step=16, label="Width")
                        height = gr.Slider(256, 1280, 480, step=16, label="Height")
                    
                    with gr.Row():
                        frames = gr.Slider(1, 200, 81, step=1, label="Frames")
                        fps = gr.Slider(1, 60, 16, step=1, label="FPS")
                    
                    with gr.Row():
                        steps = gr.Slider(1, 50, 20, step=1, label="Total Steps")
                        high_steps = gr.Slider(0, 50, 10, step=1, label="High Noise Steps")
                    
                    cfg = gr.Slider(1.0, 20.0, 1.0, step=0.1, label="CFG Scale")
                    seed = gr.Number(0, label="Seed (0 = Random)")
                
                with gr.Accordion("Optimizations", open=False):
                    use_sage = gr.Checkbox(True, label="SageAttention")
                    use_flow = gr.Checkbox(True, label="Flow Shift")
                    shift1 = gr.Slider(0, 20, 8.0, label="High Noise Shift")
                    shift2 = gr.Slider(0, 20, 8.0, label="Low Noise Shift")
                    teacache = gr.Slider(0, 1, 0.275, step=0.005, label="TeaCache")
                    
                    use_high_lora = gr.Checkbox(False, label="High Noise Speed LoRA")
                    high_str = gr.Slider(0, 2, 0.8, label="Strength")
                    use_low_lora = gr.Checkbox(False, label="Low Noise Speed LoRA")
                    low_str = gr.Slider(0, 2, 1.2, label="Strength")
                
                with gr.Accordion("RIFE Interpolation", open=False):
                    do_rife = gr.Checkbox(False, label="Enable RIFE")
                    rife_mult = gr.Slider(2, 8, 2, step=2, label="Multiplier")
                    rife_crf = gr.Slider(0, 51, 17, label="Quality (lower=better)")
                
                btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")
            
            with gr.Column(scale=55):
                output_video = gr.Video(label="Output", interactive=False, height=600)
                status = gr.Textbox(label="Status", interactive=False)
        
        btn.click(
            run_generation,
            inputs=[
                img1, img2, prompt, neg_prompt,
                width, height, frames, fps,
                steps, high_steps, cfg, seed,
                use_sage, use_flow, shift1, shift2, teacache,
                use_high_lora, high_str, use_low_lora, low_str,
                do_rife, rife_mult, rife_crf
            ],
            outputs=[output_video, status]
        )
    
    return demo


# ============================================================================
# CLI INTERFACE
# ============================================================================

def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Wan2.2 First & Last Frame to Video Generator"
    )
    
    parser.add_argument("--mode", choices=["cli", "ui"], default="cli",
                       help="Run mode: cli for command-line, ui for Gradio interface")
    
    parser.add_argument("--setup-only", action="store_true",
                       help="Only run environment setup, don't generate")
    
    # Input/Output
    parser.add_argument("-i", "--image", required=False,
                       help="Path to first frame image")
    parser.add_argument("-i2", "--image2",
                       help="Path to last frame image (optional)")
    parser.add_argument("-o", "--output-dir", default=Config.OUTPUT_DIR,
                       help="Output directory")
    
    # Prompts
    parser.add_argument("-p", "--prompt", default="a cute anime girl walking",
                       help="Positive prompt")
    parser.add_argument("-n", "--negative", default="low quality, blurry, static",
                       help="Negative prompt")
    
    # Dimensions
    parser.add_argument("-W", "--width", type=int, default=832,
                       help="Output width")
    parser.add_argument("-H", "--height", type=int, default=480,
                       help="Output height")
    parser.add_argument("-f", "--frames", type=int, default=81,
                       help="Number of frames")
    parser.add_argument("--fps", type=int, default=16,
                       help="Frames per second")
    
    # Generation params
    parser.add_argument("-s", "--steps", type=int, default=20,
                       help="Total sampling steps")
    parser.add_argument("--high-steps", type=int, default=10,
                       help="High noise model steps")
    parser.add_argument("-c", "--cfg", type=float, default=1.0,
                       help="CFG scale")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (None = random)")
    parser.add_argument("--sampler", default="uni_pc",
                       choices=["uni_pc", "euler", "dpmpp_2m"],
                       help="Sampler algorithm")
    
    # Optimizations
    parser.add_argument("--no-sage", action="store_true",
                       help="Disable SageAttention")
    parser.add_argument("--no-flow", action="store_true",
                       help="Disable flow shift")
    parser.add_argument("--teacache", type=float, default=0.275,
                       help="TeaCache threshold (0 = disable)")
    
    # Speed LoRAs
    parser.add_argument("--use-high-lora", action="store_true",
                       help="Use high noise speed LoRA")
    parser.add_argument("--use-low-lora", action="store_true",
                       help="Use low noise speed LoRA")
    
    # RIFE
    parser.add_argument("--rife", action="store_true",
                       help="Apply RIFE interpolation")
    parser.add_argument("--rife-mult", type=int, default=2, choices=[2, 4, 8],
                       help="RIFE multiplier")
    
    return parser


def main():
    """Main entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Setup environment
    models = EnvironmentSetup.setup_all()
    
    if args.setup_only:
        print("✅ Setup complete. Exiting.")
        return
    
    # Initialize generator
    generator = VideoGenerator(models)
    
    if args.mode == "ui":
        # Launch Gradio UI
        print("🌐 Launching Gradio interface...")
        demo = create_gradio_interface(generator)
        demo.launch(share=True, debug=True)
    
    else:
        # CLI mode
        if not args.image:
            print("❌ Error: --image required in CLI mode")
            parser.print_help()
            return
        
        print(f"🎬 Generating from: {args.image}")
        
        output = generator.generate(
            image_path=args.image,
            image_path2=args.image2,
            positive_prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            frames=args.frames,
            fps=args.fps,
            steps=args.steps,
            high_noise_steps=args.high_steps,
            cfg_scale=args.cfg,
            seed=args.seed,
            sampler_name=args.sampler,
            use_sage_attention=not args.no_sage,
            use_flow_shift=not args.no_flow,
            teacache_thresh=args.teacache,
            use_high_speed_lora=args.use_high_lora,
            use_low_speed_lora=args.use_low_lora
        )
        
        if args.rife and args.frames > 1:
            output = interpolate_with_rife(output, args.rife_mult)
        
        print(f"✅ Done! Output: {output}")


if __name__ == "__main__":
    main()
