#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2.2 First+Last Frame to Video Generator
Optimized for Google Colab T4/L4 with auto VRAM detection
Author: AI Fullstack Developer
"""

import os
import sys
import gc
import subprocess
import shutil
from pathlib import Path

# ============================================================================
# AUTO VRAM DETECTION & MODEL SELECTION
# ============================================================================
def detect_vram_and_select_model():
    """Tự động phát hiện VRAM và chọn quantization phù hợp"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️ Không phát hiện GPU CUDA. Sử dụng Q4_K_M (mặc định)")
            return "Q4_K_M", 15.0
        
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)
        
        print(f"🔍 Phát hiện GPU: {gpu_name}")
        print(f"💾 VRAM khả dụng: {vram_gb:.2f} GB")
        
        # T4: ~15GB → Q4_K_M, L4: ~24GB → Q6_K
        if vram_gb >= 22:
            quant = "Q6_K"
            print("✅ Chọn Q6_K (Chất lượng cao)")
        else:
            quant = "Q4_K_M"
            print("✅ Chọn Q4_K_M (Tốc độ cao)")
        
        return quant, vram_gb
    except Exception as e:
        print(f"⚠️ Lỗi phát hiện VRAM: {e}. Dùng Q4_K_M")
        return "Q4_K_M", 15.0


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
def setup_environment():
    """Cài đặt môi trường và dependencies"""
    print("=" * 60)
    print("🚀 BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG")
    print("=" * 60)
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 1. Install PyTorch
    print("\n[1/8] Cài đặt PyTorch...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "torch==2.6.0", "torchvision==0.21.0", "--upgrade"
    ], check=True)
    
    # 2. Install core dependencies
    print("[2/8] Cài đặt thư viện AI core...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "torchsde", "einops", "diffusers", "accelerate",
        "xformers==0.0.29.post2", "triton==3.2.0", "sageattention==1.0.6"
    ], check=True)
    
    # 3. Install image/video processing
    print("[3/8] Cài đặt thư viện xử lý media...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "av", "spandrel", "albumentations", "insightface",
        "onnx", "opencv-python", "segment_anything", "ultralytics",
        "onnxruntime", "onnxruntime-gpu", "imageio", "imageio-ffmpeg"
    ], check=True)
    
    # 4. Clone ComfyUI
    print("[4/8] Clone ComfyUI...")
    if not Path("/content/ComfyUI").exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "--branch", "ComfyUI_v0.3.47",
            "https://github.com/Isi-dev/ComfyUI"
        ], check=True)
    
    # 5. Clone custom nodes
    print("[5/8] Clone custom nodes...")
    custom_nodes_dir = Path("/content/ComfyUI/custom_nodes")
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    
    if not (custom_nodes_dir / "ComfyUI_GGUF").exists():
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", "forHidream",
            "https://github.com/Isi-dev/ComfyUI_GGUF.git",
            str(custom_nodes_dir / "ComfyUI_GGUF")
        ], check=True)
    
    if not (custom_nodes_dir / "ComfyUI_KJNodes").exists():
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", "kjnv1.1.3",
            "https://github.com/Isi-dev/ComfyUI_KJNodes.git",
            str(custom_nodes_dir / "ComfyUI_KJNodes")
        ], check=True)
    
    # 6. Install requirements for custom nodes
    print("[6/8] Cài đặt requirements cho custom nodes...")
    for node_dir in ["ComfyUI_GGUF", "ComfyUI_KJNodes"]:
        req_file = custom_nodes_dir / node_dir / "requirements.txt"
        if req_file.exists():
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)
            ], check=True)
    
    # 7. Setup RIFE for frame interpolation
    print("[7/8] Cài đặt RIFE (Frame Interpolation)...")
    if not Path("/content/Practical-RIFE").exists():
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Isi-dev/Practical-RIFE"
        ], check=True)
    
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/rk-exxec/scikit-video.git@numpy_deprecation"
    ], check=True)
    
    # Download RIFE models
    rife_dir = Path("/content/Practical-RIFE/train_log")
    rife_dir.mkdir(parents=True, exist_ok=True)
    
    rife_files = {
        "IFNet_HDv3.py": "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/IFNet_HDv3.py",
        "RIFE_HDv3.py": "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/RIFE_HDv3.py",
        "refine.py": "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/refine.py",
        "flownet.pkl": "https://huggingface.co/Isi99999/Frame_Interpolation_Models/resolve/main/4.25/train_log/flownet.pkl"
    }
    
    for filename, url in rife_files.items():
        subprocess.run([
            "wget", "-q", url, "-O", str(rife_dir / filename)
        ], check=True)
    
    # 8. Install aria2 and ffmpeg
    print("[8/8] Cài đặt aria2 và ffmpeg...")
    subprocess.run(["apt", "-y", "install", "-qq", "aria2", "ffmpeg"], check=True)
    
    # Install Gradio
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "gradio"
    ], check=True)
    
    print("\n✅ Hoàn tất cài đặt môi trường!")


# ============================================================================
# MODEL DOWNLOAD
# ============================================================================
def download_models(model_quant):
    """Download models với quantization đã chọn"""
    print("\n" + "=" * 60)
    print("📥 TẢI MODELS")
    print("=" * 60)
    
    models_dir = {
        "diffusion": Path("/content/ComfyUI/models/diffusion_models"),
        "text_encoder": Path("/content/ComfyUI/models/text_encoders"),
        "vae": Path("/content/ComfyUI/models/vae"),
        "clip_vision": Path("/content/ComfyUI/models/clip_vision"),
        "loras": Path("/content/ComfyUI/models/loras")
    }
    
    for d in models_dir.values():
        d.mkdir(parents=True, exist_ok=True)
    
    def aria_download(url, dest_dir, filename=None):
        """Download với aria2c"""
        if filename is None:
            filename = url.split('/')[-1].split('?')[0]
        
        dest_path = dest_dir / filename
        if dest_path.exists():
            print(f"  ⏭️ Đã có: {filename}")
            return filename
        
        print(f"  ⬇️ Đang tải: {filename}")
        subprocess.run([
            "aria2c", "--console-log-level=error",
            "-c", "-x", "16", "-s", "16", "-k", "1M",
            "-d", str(dest_dir), "-o", filename, url
        ], check=True)
        return filename
    
    # 1. Main diffusion models
    print("\n[1/5] Tải Wan2.2 Diffusion Models...")
    model_urls = {
        "Q4_K_M": {
            "high": "https://huggingface.co/Isi99999/Wan2.2BasedModels/resolve/main/wan2.2_i2v_high_noise_14B_Q4_K_M.gguf",
            "low": "https://huggingface.co/Isi99999/Wan2.2BasedModels/resolve/main/wan2.2_i2v_low_noise_14B_Q4_K_M.gguf"
        },
        "Q6_K": {
            "high": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_high_noise_14B_Q6_K.gguf",
            "low": "https://huggingface.co/bullerwins/Wan2.2-I2V-A14B-GGUF/resolve/main/wan2.2_i2v_low_noise_14B_Q6_K.gguf"
        }
    }
    
    urls = model_urls[model_quant]
    dit_high = aria_download(urls["high"], models_dir["diffusion"])
    dit_low = aria_download(urls["low"], models_dir["diffusion"])
    
    # 2. Text encoder
    print("\n[2/5] Tải Text Encoder...")
    te_file = aria_download(
        "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        models_dir["text_encoder"]
    )
    
    # 3. VAE
    print("\n[3/5] Tải VAE...")
    vae_file = aria_download(
        "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/wan_2.1_vae.safetensors",
        models_dir["vae"]
    )
    
    # 4. CLIP Vision
    print("\n[4/5] Tải CLIP Vision...")
    clip_file = aria_download(
        "https://huggingface.co/Isi99999/Wan_Extras/resolve/main/clip_vision_h.safetensors",
        models_dir["clip_vision"]
    )
    
    # 5. Speed LoRAs
    print("\n[5/5] Tải Speed LoRAs...")
    lora_high = aria_download(
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors",
        models_dir["loras"]
    )
    lora_low = aria_download(
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22-Lightning/Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors",
        models_dir["loras"]
    )
    
    # Motion LoRAs
    print("  ⬇️ Tải Motion LoRAs...")
    aria_download(
        "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/walking%20to%20viewers_Wan.safetensors",
        models_dir["loras"], "walking_to_viewers_Wan.safetensors"
    )
    aria_download(
        "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/walking_from_behind.safetensors",
        models_dir["loras"]
    )
    aria_download(
        "https://huggingface.co/Isi99999/Wan2.1_14B-480p_I2V_LoRAs/resolve/main/b3ll13-d8nc3r.safetensors",
        models_dir["loras"]
    )
    
    print("\n✅ Hoàn tất tải models!")
    return {
        "dit_high": dit_high,
        "dit_low": dit_low,
        "text_encoder": te_file,
        "vae": vae_file,
        "clip_vision": clip_file,
        "lora_high": lora_high,
        "lora_low": lora_low
    }


# ============================================================================
# VIDEO GENERATION CORE
# ============================================================================
def init_comfy_nodes():
    """Khởi tạo các nodes của ComfyUI"""
    sys.path.insert(0, '/content/ComfyUI')
    
    from nodes import (
        CLIPLoader, CLIPTextEncode, VAEDecode, VAELoader,
        KSamplerAdvanced, LoadImage, LoraLoaderModelOnly,
        ImageScale, CLIPVisionLoader, CLIPVisionEncode
    )
    from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
    from custom_nodes.ComfyUI_KJNodes.nodes.model_optimization_nodes import (
        WanVideoTeaCacheKJ, PathchSageAttentionKJ, WanVideoNAG
    )
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
    from comfy_extras.nodes_wan import WanFirstLastFrameToVideo
    
    return {
        "unet_loader": UnetLoaderGGUF(),
        "clip_loader": CLIPLoader(),
        "clip_encode": CLIPTextEncode(),
        "vae_loader": VAELoader(),
        "vae_decode": VAEDecode(),
        "clip_vision_loader": CLIPVisionLoader(),
        "clip_vision_encode": CLIPVisionEncode(),
        "load_image": LoadImage(),
        "image_scale": ImageScale(),
        "wan_encoder": WanFirstLastFrameToVideo(),
        "ksampler": KSamplerAdvanced(),
        "lora_loader": LoraLoaderModelOnly(),
        "sage_attention": PathchSageAttentionKJ(),
        "nag": WanVideoNAG(),
        "teacache": WanVideoTeaCacheKJ(),
        "model_sampling": ModelSamplingSD3()
    }


def generate_video_core(
    nodes, models,
    img1_path, img2_path,
    prompt, neg_prompt,
    width, height, frames, fps,
    steps, high_noise_steps, cfg, seed,
    sampler, scheduler,
    use_high_speed, high_speed_str,
    use_low_speed, low_speed_str,
    use_sage, teacache_thresh,
    flow_shift, flow_shift2
):
    """
    Core video generation function
    """
    import torch
    import numpy as np
    import imageio
    import datetime
    
    with torch.inference_mode():
        # 1. Load và encode text
        print("📝 Encoding prompts...")
        clip = nodes["clip_loader"].load_clip(models["text_encoder"], "wan", "default")[0]
        pos_cond = nodes["clip_encode"].encode(clip, prompt)[0]
        neg_cond = nodes["clip_encode"].encode(clip, neg_prompt)[0]
        del clip
        torch.cuda.empty_cache()
        
        # 2. Load images
        print("🖼️ Loading images...")
        clip_vision = nodes["clip_vision_loader"].load_clip(models["clip_vision"])[0]
        
        img1 = nodes["load_image"].load_image(img1_path)[0]
        img1 = nodes["image_scale"].upscale(img1, "lanczos", width, height, "disabled")[0]
        clip_vis1 = nodes["clip_vision_encode"].encode(clip_vision, img1, "none")[0]
        
        img2, clip_vis2 = None, None
        if img2_path:
            img2 = nodes["load_image"].load_image(img2_path)[0]
            img2 = nodes["image_scale"].upscale(img2, "lanczos", width, height, "disabled")[0]
            clip_vis2 = nodes["clip_vision_encode"].encode(clip_vision, img2, "none")[0]
        
        del clip_vision
        torch.cuda.empty_cache()
        
        # 3. Load VAE và encode
        print("🔄 Encoding to latent space...")
        vae = nodes["vae_loader"].load_vae(models["vae"])[0]
        pos_out, neg_out, latent = nodes["wan_encoder"].encode(
            pos_cond, neg_cond, vae, width, height, frames, 1,
            img1, img2, clip_vis1, clip_vis2
        )
        
        # 4. High noise pass
        print("⚡ High noise generation...")
        model = nodes["unet_loader"].load_unet(models["dit_high"])[0]
        model = nodes["nag"].patch(model, neg_cond, 11.0, 0.25, 2.5)[0]
        model = nodes["model_sampling"].patch(model, flow_shift)[0]
        
        if use_high_speed:
            model = nodes["lora_loader"].load_lora_model_only(
                model, models["lora_high"], high_speed_str
            )[0]
        
        if use_sage:
            model = nodes["sage_attention"].patch(model, "auto")[0]
        
        if teacache_thresh > 0:
            model = nodes["teacache"].patch_teacache(
                model, teacache_thresh, 0.1, 1.0, "main_device", "14B"
            )[0]
        
        sampled = nodes["ksampler"].sample(
            model=model,
            add_noise="enable",
            noise_seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            positive=pos_out,
            negative=neg_out,
            latent_image=latent,
            start_at_step=0,
            end_at_step=high_noise_steps,
            return_with_leftover_noise="enable"
        )[0]
        
        del model
        torch.cuda.empty_cache()
        
        # 5. Low noise pass
        print("✨ Low noise refinement...")
        model = nodes["unet_loader"].load_unet(models["dit_low"])[0]
        model = nodes["model_sampling"].patch(model, flow_shift2)[0]
        
        if use_low_speed:
            model = nodes["lora_loader"].load_lora_model_only(
                model, models["lora_low"], low_speed_str
            )[0]
        
        if use_sage:
            model = nodes["sage_attention"].patch(model, "auto")[0]
        
        if teacache_thresh > 0:
            model = nodes["teacache"].patch_teacache(
                model, teacache_thresh, 0.1, 1.0, "main_device", "14B"
            )[0]
        
        sampled = nodes["ksampler"].sample(
            model=model,
            add_noise="disable",
            noise_seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            positive=pos_out,
            negative=neg_out,
            latent_image=sampled,
            start_at_step=high_noise_steps,
            end_at_step=10000,
            return_with_leftover_noise="disable"
        )[0]
        
        del model
        torch.cuda.empty_cache()
        
        # 6. Decode
        print("🎬 Decoding video...")
        decoded = nodes["vae_decode"].decode(vae, sampled)[0]
        del vae
        torch.cuda.empty_cache()
        
        # 7. Save
        output_dir = Path("/content/ComfyUI/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"output_{timestamp}.mp4"
        
        frames_np = [(f.cpu().numpy() * 255).astype(np.uint8) for f in decoded]
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames_np:
                writer.append_data(frame)
        
        print(f"✅ Saved: {output_path}")
        return str(output_path)


def run_rife_interpolation(input_video, multiplier, crf):
    """Chạy RIFE frame interpolation"""
    print(f"🔄 Running RIFE {multiplier}x interpolation...")
    
    cwd = os.getcwd()
    try:
        os.chdir("/content/Practical-RIFE")
        
        cmd = [
            sys.executable, "inference_video.py",
            f"--multi={int(multiplier)}",
            f"--fps=30",
            f"--video={input_video}",
            "--scale=1"
        ]
        
        env = os.environ.copy()
        env["XDG_RUNTIME_DIR"] = "/tmp"
        env["SDL_AUDIODRIVER"] = "dummy"
        
        subprocess.run(cmd, env=env, check=True)
        
        # Find output and compress
        import glob
        import random
        video_files = glob.glob("/content/ComfyUI/output/*.mp4")
        latest = max(video_files, key=os.path.getctime)
        
        output_path = f"/content/ComfyUI/output/rife_{random.randint(0,9999)}.mp4"
        subprocess.run([
            "ffmpeg", "-i", latest,
            "-vcodec", "libx264", "-crf", str(int(crf)),
            "-preset", "fast", output_path,
            "-loglevel", "error", "-y"
        ], check=True)
        
        return output_path
    finally:
        os.chdir(cwd)


# ============================================================================
# GRADIO INTERFACE
# ============================================================================
def create_gradio_app(nodes, models):
    """Tạo giao diện Gradio"""
    import gradio as gr
    import random
    from PIL import Image
    
    def gradio_generate(
        img1, img2,
        prompt, neg_prompt,
        width, height, frames, fps,
        steps, high_noise_steps, cfg, seed,
        sampler, scheduler,
        use_high_speed, high_speed_str,
        use_low_speed, low_speed_str,
        use_sage, teacache_thresh,
        do_rife, rife_mult, rife_crf
    ):
        try:
            if img1 is None:
                return None, "❌ Cần upload ảnh đầu!"
            
            # Save images
            os.makedirs("/content/ComfyUI/input", exist_ok=True)
            img1_path = "/content/ComfyUI/input/img1.png"
            img1.save(img1_path)
            
            img2_path = None
            if img2 is not None:
                img2_path = "/content/ComfyUI/input/img2.png"
                img2.save(img2_path)
            
            # Random seed
            if seed == 0:
                seed = random.randint(0, 2**32 - 1)
            
            # Generate
            output_video = generate_video_core(
                nodes, models,
                img1_path, img2_path,
                prompt, neg_prompt,
                int(width), int(height), int(frames), int(fps),
                int(steps), int(high_noise_steps), cfg, seed,
                sampler, scheduler,
                use_high_speed, high_speed_str,
                use_low_speed, low_speed_str,
                use_sage, teacache_thresh,
                8.0, 8.0  # flow shifts
            )
            
            status = f"✅ Hoàn tất! Seed: {seed}"
            
            # RIFE interpolation
            if do_rife:
                output_video = run_rife_interpolation(output_video, rife_mult, rife_crf)
                status += f" | RIFE {rife_mult}x"
            
            return output_video, status
            
        except Exception as e:
            import traceback
            return None, f"❌ Lỗi: {str(e)}\n{traceback.format_exc()}"
    
    # UI Layout
    with gr.Blocks(title="Wan2.2 Optimized", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬 Wan2.2 Video Generator (Auto-Optimized)")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input Images")
                img1 = gr.Image(label="Ảnh đầu (Required)", type="pil", height=280)
                img2 = gr.Image(label="Ảnh cuối (Optional)", type="pil", height=280)
                
                gr.Markdown("### 📝 Prompts")
                prompt = gr.Textbox(
                    label="Positive Prompt",
                    placeholder="Mô tả chuyển động...",
                    lines=2,
                    value="a cute anime girl walking towards camera"
                )
                neg_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    value="static, blurry, low quality, distorted"
                )
                
                with gr.Accordion("⚙️ Settings", open=True):
                    with gr.Row():
                        width = gr.Slider(256, 1280, 832, step=16, label="Width")
                        height = gr.Slider(256, 1280, 480, step=16, label="Height")
                    
                    with gr.Row():
                        frames = gr.Slider(1, 200, 81, step=1, label="Frames")
                        fps = gr.Slider(1, 60, 16, step=1, label="FPS")
                    
                    with gr.Row():
                        steps = gr.Slider(1, 50, 20, step=1, label="Steps")
                        high_noise_steps = gr.Slider(0, 50, 10, step=1, label="High Noise Steps")
                    
                    cfg = gr.Slider(1.0, 20.0, 1.0, step=0.1, label="CFG Scale")
                    seed = gr.Number(0, label="Seed (0=Random)")
                    
                    sampler = gr.Dropdown(
                        ["uni_pc", "euler", "dpmpp_2m"],
                        value="uni_pc",
                        label="Sampler"
                    )
                    scheduler = gr.Dropdown(
                        ["simple", "karras", "normal"],
                        value="simple",
                        label="Scheduler"
                    )
                
                with gr.Accordion("🚀 Optimization", open=True):
                    with gr.Row():
                        use_high = gr.Checkbox(True, label="High Speed LoRA")
                        high_str = gr.Slider(0, 2, 0.8, step=0.1, label="Strength")
                    
                    with gr.Row():
                        use_low = gr.Checkbox(True, label="Low Speed LoRA")
                        low_str = gr.Slider(0, 2, 1.2, step=0.1, label="Strength")
                    
                    use_sage = gr.Checkbox(True, label="SageAttention (Save VRAM)")
                    teacache = gr.Slider(0, 1, 0.25, step=0.05, label="TeaCache (0=Off)")
                
                with gr.Accordion("🎞️ RIFE Interpolation", open=False):
                    do_rife = gr.Checkbox(False, label="Enable RIFE")
                    with gr.Row():
                        rife_mult = gr.Slider(2, 8, 2, step=2, label="Multiplier")
                        rife_crf = gr.Slider(0, 51, 17, step=1, label="Quality (lower=better)")
                
                btn = gr.Button("🚀 GENERATE VIDEO", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_video = gr.Video(label="Output", height=600)
                status = gr.Textbox(label="Status", interactive=False)
        
        btn.click(
            gradio_generate,
            inputs=[
                img1, img2, prompt, neg_prompt,
                width, height, frames, fps,
                steps, high_noise_steps, cfg, seed,
                sampler, scheduler,
                use_high, high_str, use_low, low_str,
                use_sage, teacache,
                do_rife, rife_mult, rife_crf
            ],
            outputs=[output_video, status]
        )
    
    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution flow"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           Wan2.2 Video Generator - Auto Setup               ║
    ║              Optimized for Google Colab T4/L4                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 1. Detect VRAM and select model
    model_quant, vram = detect_vram_and_select_model()
    
    # 2. Setup environment
    setup_environment()
    
    # 3. Download models
    models = download_models(model_quant)
    
    # 4. Initialize ComfyUI nodes
    print("\n" + "=" * 60)
    print("🔧 KHỞI TẠO COMFYUI NODES")
    print("=" * 60)
    nodes = init_comfy_nodes()
    print("✅ Nodes initialized!")
    
    # 5. Create and launch Gradio
    print("\n" + "=" * 60)
    print("🌐 KHỞI CHẠY GRADIO INTERFACE")
    print("=" * 60)
    app = create_gradio_app(nodes, models)
    app.launch(
        share=True,
        debug=False,
        show_error=True,
        inline=False
    )


if __name__ == "__main__":
    main()
