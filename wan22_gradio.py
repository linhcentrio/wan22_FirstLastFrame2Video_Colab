#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.2 First & Last Frame to Video Generator
Optimized for Colab T4/L4 GPU with Auto VRAM Detection
Single-command setup & run
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def setup_environment():
    """Cài đặt môi trường và dependencies"""
    print("🚀 Bắt đầu cài đặt môi trường...")
    
    # Suppress warnings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    commands = [
        # Install PyTorch
        "pip install -q torch==2.6.0 torchvision==0.21.0",
        
        # Install core dependencies
        "pip install -q torchsde einops diffusers accelerate xformers==0.0.29.post2 triton==3.2.0 sageattention==1.0.6",
        "pip install -q av spandrel albumentations insightface onnx opencv-python segment_anything ultralytics onnxruntime onnxruntime-gpu",
        
        # Clone ComfyUI
        "git clone -q --depth 1 --branch ComfyUI_v0.3.47 https://github.com/Isi-dev/ComfyUI /content/ComfyUI",
        
        # Clone custom nodes
        "git clone -q --depth 1 --branch forHidream https://github.com/Isi-dev/ComfyUI_GGUF.git /content/ComfyUI/custom_nodes/ComfyUI_GGUF",
        "git clone -q --depth 1 --branch kjnv1.1.3 https://github.com/Isi-dev/ComfyUI_KJNodes.git /content/ComfyUI/custom_nodes/ComfyUI_KJNodes",
        
        # Install custom node requirements
        "pip install -q -r /content/ComfyUI/custom_nodes/ComfyUI_GGUF/requirements.txt",
        "pip install -q -r /content/ComfyUI/custom_nodes/ComfyUI_KJNodes/requirements.txt",
        
        # Clone RIFE for interpolation
        "git clone -q --depth 1 https://github.com/Isi-dev/Practical-RIFE /content/Practical-RIFE",
        "pip install -q git+https://github.com/rk-exxec/scikit-video.git@numpy_deprecation",
        
        # Install ffmpeg & aria2c
        "apt-get -qq update && apt-get -qq install -y aria2 ffmpeg > /dev/null 2>&1",
        
        # Install Gradio
        "pip install -q gradio"
    ]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Download RIFE models
    os.makedirs("/content/Practical-RIFE/train_log", exist_ok=True)
    rife_files = [
        "IFNet_HDv3.py", "RIFE_HDv3.py", "refine.py", "flownet.pkl"
    ]
    base_url = "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/"
    for fname in rife_files:
        subprocess.run(
            f"wget -q {base_url}{fname} -O /content/Practical-RIFE/train_log/{fname}",
            shell=True
        )
    
    print("✅ Môi trường đã sẵn sàng!")

def detect_vram():
    """Phát hiện VRAM và chọn model phù hợp"""
    import torch
    if not torch.cuda.is_available():
        return "Q4_K_M"
    
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🔍 Phát hiện VRAM: {vram_gb:.1f} GB")
    
    if vram_gb >= 20:  # L4 or better
        print("✅ Sử dụng model Q6_K (Chất lượng cao)")
        return "Q6_K"
    else:  # T4
        print("✅ Sử dụng model Q4_K_M (Tối ưu tốc độ)")
        return "Q4_K_M"

def download_models(quant="Q4_K_M"):
    """Download models với aria2c"""
    print(f"📥 Đang tải models ({quant})...")
    
    def aria_download(url, dest_dir, filename=None):
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        fname = filename or url.split('/')[-1]
        cmd = f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M '{url}' -d '{dest_dir}' -o '{fname}'"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        return fname
    
    # Model URLs
    if quant == "Q6_K":
        high_noise = "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q6_K.gguf"
        low_noise = "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q6_K.gguf"
    else:
        high_noise = "https://huggingface.co/Isi99999/Wan2.2BasedModels/resolve/main/wan2.2_i2v_high_noise_14B_Q4_K_M.gguf"
        low_noise = "https://huggingface.co/Isi99999/Wan2.2BasedModels/resolve/main/wan2.2_i2v_low_noise_14B_Q4_K_M.gguf"
    
    # Download main models
    dit_high = aria_download(high_noise, "/content/ComfyUI/models/diffusion_models")
    dit_low = aria_download(low_noise, "/content/ComfyUI/models/diffusion_models")
    
    # Download essentials
    aria_download(
        "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "/content/ComfyUI/models/text_encoders"
    )
    aria_download(
        "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/wan_2.1_vae.safetensors",
        "/content/ComfyUI/models/vae"
    )
    aria_download(
        "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/clip_vision_h.safetensors",
        "/content/ComfyUI/models/clip_vision"
    )
    
    # Download LoRAs
    aria_download(
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors",
        "/content/ComfyUI/models/loras"
    )
    aria_download(
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors",
        "/content/ComfyUI/models/loras"
    )
    
    print("✅ Models đã tải xong!")
    return dit_high, dit_low

# ==================== MAIN GENERATION CODE ====================

sys.path.insert(0, '/content/ComfyUI')

import torch
import numpy as np
import cv2
from PIL import Image
import gc
import random
import imageio
import gradio as gr
from datetime import datetime

# Import ComfyUI nodes
from comfy import model_management
from nodes import CLIPLoader, CLIPTextEncode, VAEDecode, VAELoader, KSamplerAdvanced, LoadImage, ImageScale
from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
from custom_nodes.ComfyUI_KJNodes.nodes.model_optimization_nodes import (
    WanVideoTeaCacheKJ, PathchSageAttentionKJ, WanVideoNAG
)
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_wan import WanFirstLastFrameToVideo
from nodes import CLIPVisionLoader, CLIPVisionEncode, LoraLoaderModelOnly

# Global model paths (will be set after download)
MODEL_HIGH = None
MODEL_LOW = None

def clear_memory():
    """Dọn dẹp bộ nhớ"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def save_video(frames, output_path, fps=16):
    """Lưu video ra file MP4"""
    frames_uint8 = [(f.cpu().numpy() * 255).astype(np.uint8) for f in frames]
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames_uint8:
            writer.append_data(frame)
    return output_path

def generate_video(
    img1, img2,
    prompt, negative_prompt,
    width, height, frames, fps,
    steps, high_noise_steps, cfg,
    seed, use_speedup, teacache
):
    """Hàm tạo video chính"""
    
    if img1 is None:
        return None, "❌ Vui lòng tải ảnh đầu!"
    
    # Random seed nếu = 0
    if seed == 0:
        seed = random.randint(0, 2**32-1)
    
    print(f"🎬 Bắt đầu tạo video | Seed: {seed}")
    
    with torch.inference_mode():
        # 1. Load Text Encoder
        print("📝 Loading CLIP...")
        clip_loader = CLIPLoader()
        clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
        
        clip_encode = CLIPTextEncode()
        positive = clip_encode.encode(clip, prompt)[0]
        negative = clip_encode.encode(clip, negative_prompt)[0]
        del clip
        clear_memory()
        
        # 2. Load CLIP Vision
        print("👁️ Loading CLIP Vision...")
        clip_vision_loader = CLIPVisionLoader()
        clip_vision = clip_vision_loader.load_clip("clip_vision_h.safetensors")[0]
        
        clip_vision_encode = CLIPVisionEncode()
        
        # Prepare images
        os.makedirs("/content/ComfyUI/input", exist_ok=True)
        img1_path = "/content/ComfyUI/input/temp_img1.png"
        img1.save(img1_path)
        
        load_image_node = LoadImage()
        loaded_img1 = load_image_node.load_image("temp_img1.png")[0]
        
        # Scale image
        scaler = ImageScale()
        loaded_img1 = scaler.upscale(loaded_img1, "lanczos", width, height, "disabled")[0]
        
        clip_vision_out1 = clip_vision_encode.encode(clip_vision, loaded_img1, "none")[0]
        
        # Handle second image
        loaded_img2 = None
        clip_vision_out2 = None
        if img2 is not None:
            img2_path = "/content/ComfyUI/input/temp_img2.png"
            img2.save(img2_path)
            loaded_img2 = load_image_node.load_image("temp_img2.png")[0]
            loaded_img2 = scaler.upscale(loaded_img2, "lanczos", width, height, "disabled")[0]
            clip_vision_out2 = clip_vision_encode.encode(clip_vision, loaded_img2, "none")[0]
        
        del clip_vision
        clear_memory()
        
        # 3. Load VAE
        print("🎨 Loading VAE...")
        vae_loader = VAELoader()
        vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]
        
        # Encode to latent
        wan_encoder = WanFirstLastFrameToVideo()
        pos_out, neg_out, latent = wan_encoder.encode(
            positive, negative, vae, width, height, frames, 1,
            loaded_img1, loaded_img2, clip_vision_out1, clip_vision_out2
        )
        
        # 4. Load High Noise Model
        print("🧠 Loading High Noise Model...")
        unet_loader = UnetLoaderGGUF()
        model = unet_loader.load_unet(MODEL_HIGH)[0]
        
        # Apply optimizations
        nag = WanVideoNAG()
        model = nag.patch(model, negative, 11.0, 0.25, 2.5)[0]
        
        model_sampling = ModelSamplingSD3()
        model = model_sampling.patch(model, 8.0)[0]
        
        # Load speedup LoRA if enabled
        if use_speedup:
            lora_loader = LoraLoaderModelOnly()
            lora_name = "Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors"
            model = lora_loader.load_lora_model_only(model, lora_name, 0.8)[0]
        
        # Sage Attention
        sage = PathchSageAttentionKJ()
        model = sage.patch(model, "auto")[0]
        
        # TeaCache
        if teacache > 0:
            teacache_node = WanVideoTeaCacheKJ()
            model = teacache_node.patch_teacache(model, teacache, 0.1, 1.0, "main_device", "14B")[0]
        
        # Sample with high noise model
        print("⚡ Sampling (High Noise)...")
        sampler = KSamplerAdvanced()
        latent_hn = sampler.sample(
            model=model, add_noise="enable", noise_seed=seed,
            steps=steps, cfg=cfg, sampler_name="uni_pc", scheduler="simple",
            positive=pos_out, negative=neg_out, latent_image=latent,
            start_at_step=0, end_at_step=high_noise_steps,
            return_with_leftover_noise="enable"
        )[0]
        
        del model
        clear_memory()
        
        # 5. Load Low Noise Model
        print("🧠 Loading Low Noise Model...")
        model = unet_loader.load_unet(MODEL_LOW)[0]
        model = model_sampling.patch(model, 8.0)[0]
        
        if use_speedup:
            lora_name = "Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors"
            model = lora_loader.load_lora_model_only(model, lora_name, 1.2)[0]
        
        model = sage.patch(model, "auto")[0]
        
        if teacache > 0:
            model = teacache_node.patch_teacache(model, teacache, 0.1, 1.0, "main_device", "14B")[0]
        
        # Final sampling
        print("⚡ Sampling (Low Noise)...")
        latent_final = sampler.sample(
            model=model, add_noise="disable", noise_seed=seed,
            steps=steps, cfg=cfg, sampler_name="uni_pc", scheduler="simple",
            positive=pos_out, negative=neg_out, latent_image=latent_hn,
            start_at_step=high_noise_steps, end_at_step=10000,
            return_with_leftover_noise="disable"
        )[0]
        
        del model
        clear_memory()
        
        # 6. Decode
        print("🎞️ Decoding...")
        vae_decode = VAEDecode()
        decoded = vae_decode.decode(vae, latent_final)[0]
        del vae
        clear_memory()
        
        # 7. Save video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/content/ComfyUI/output/video_{timestamp}.mp4"
        os.makedirs("/content/ComfyUI/output", exist_ok=True)
        save_video(decoded, output_path, fps)
        
        print(f"✅ Hoàn tất! Seed: {seed}")
        return output_path, f"✅ Thành công! Seed: {seed}"

# ==================== GRADIO UI ====================

def create_ui():
    """Tạo giao diện Gradio"""
    
    with gr.Blocks(title="Wan2.2 Video Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 Wan2.2 First & Last Frame to Video")
        gr.Markdown("*Tối ưu cho Colab T4/L4 GPU*")
        
        with gr.Row():
            # Input column
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Hình ảnh")
                img1 = gr.Image(label="Ảnh đầu (Bắt buộc)", type="pil", height=280)
                img2 = gr.Image(label="Ảnh cuối (Tùy chọn)", type="pil", height=280)
                
                gr.Markdown("### ✍️ Mô tả")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Mô tả chuyển động...",
                    lines=3,
                    value="a woman walking towards camera, cinematic"
                )
                neg_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    value="static, blurry, low quality, distorted"
                )
                
                with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                    with gr.Row():
                        width = gr.Slider(256, 1280, 832, step=16, label="Rộng")
                        height = gr.Slider(256, 1280, 480, step=16, label="Cao")
                    
                    with gr.Row():
                        frames = gr.Slider(1, 200, 81, step=1, label="Frames")
                        fps = gr.Slider(1, 60, 16, step=1, label="FPS")
                    
                    with gr.Row():
                        steps = gr.Slider(1, 50, 20, step=1, label="Tổng bước")
                        high_noise_steps = gr.Slider(0, 50, 10, step=1, label="Bước nhiễu cao")
                    
                    with gr.Row():
                        cfg = gr.Slider(1.0, 20.0, 1.0, step=0.1, label="CFG")
                        seed = gr.Number(0, label="Seed (0=Random)")
                    
                    with gr.Row():
                        use_speedup = gr.Checkbox(True, label="🚀 Tăng tốc (Lightning LoRA)")
                        teacache = gr.Slider(0, 1, 0.25, step=0.05, label="TeaCache")
                
                btn = gr.Button("🎬 TẠO VIDEO", variant="primary", size="lg")
            
            # Output column
            with gr.Column(scale=1):
                output_video = gr.Video(label="Kết quả", height=600)
                status = gr.Textbox(label="Trạng thái", interactive=False)
        
        btn.click(
            fn=generate_video,
            inputs=[
                img1, img2, prompt, neg_prompt,
                width, height, frames, fps,
                steps, high_noise_steps, cfg,
                seed, use_speedup, teacache
            ],
            outputs=[output_video, status]
        )
    
    return demo

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Setup environment
    setup_environment()
    
    # Detect VRAM and download models
    quant = detect_vram()
    MODEL_HIGH, MODEL_LOW = download_models(quant)
    
    # Launch Gradio
    print("\n🎉 Khởi động Gradio UI...")
    demo = create_ui()
    demo.launch(share=True, debug=False)
    
    # Cleanup (optional - uncomment if needed)
    # print("\n🧹 Dọn dẹp files setup...")
    # if os.path.exists("/content/setup_temp"):
    #     shutil.rmtree("/content/setup_temp")
