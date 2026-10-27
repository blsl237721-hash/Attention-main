import os
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F_t

# 1. 关键：只导入官方库，绝不导入那个报错的自定义文件
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from accelerate.utils import set_seed
from pipeline_sd import ADPipeline
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# ================= 🔧 配置区域 (修改这里) =================

# 1. 官方底模路径 (纯净版 SD 1.5)
BASE_MODEL_PATH = "./models/stable-diffusion-v1-5"

# 2. 输入图片路径
STYLE_IMAGE_PATH = "./f/aircraft_11.jpg"   
CONTENT_IMAGE_PATH = "./n/marine-debris-aris3k-9.png"      
MASK_IMAGE_PATH = "./2.jpg"   # 你的黑白扇形掩膜

# 3. 输出设置
OUTPUT_DIR = "./standard_masked_output"
OUTPUT_FILENAME = "result_masked_standard.png"

# 4. 超参数
width = 512
height = 512
seed = 42
num_inference_steps = 200
start_layer, end_layer = 10, 16 

# ========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
final_save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

print(f"=== 标准风格迁移 + 掩膜去噪 ===")
print(f"风格: {STYLE_IMAGE_PATH}")
print(f"内容: {CONTENT_IMAGE_PATH}")
print(f"掩膜: {MASK_IMAGE_PATH}")

# ==========================================
# 1. 加载原生模型 (绝对不报错)
# ==========================================
print("\n🔄 加载官方 SD 1.5 模型...")
scheduler = DDIMScheduler.from_pretrained(BASE_MODEL_PATH, subfolder="scheduler", local_files_only=True)
vae_model = AutoencoderKL.from_pretrained(BASE_MODEL_PATH, subfolder="vae", local_files_only=True)

# 【核心】加载官方 UNet (Standard)
print(f"👉 加载官方 UNet...")
unet = UNet2DConditionModel.from_pretrained(
    BASE_MODEL_PATH, 
    subfolder="unet", 
    torch_dtype=torch.float16,
    local_files_only=True
)

# 组装 Pipeline
pipe = ADPipeline.from_pretrained(
    BASE_MODEL_PATH,
    unet=unet,
    scheduler=scheduler, 
    vae=vae_model,
    safety_checker=None, 
    torch_dtype=torch.float16,
    local_files_only=True
)
pipe.classifier = pipe.unet
pipe.to("cuda")
set_seed(seed)

# ==========================================
# 2. 执行推理
# ==========================================
if not os.path.exists(STYLE_IMAGE_PATH) or not os.path.exists(CONTENT_IMAGE_PATH):
    raise FileNotFoundError("输入图片不存在！")

style_image = load_image(STYLE_IMAGE_PATH, size=(width, height))
content_image = load_image(CONTENT_IMAGE_PATH, size=(width, height))
controller = Controller(self_layers=(start_layer, end_layer))

print("🚀 开始生成 (会有背景泄露)...")
start_time = time.time()

# 生成原始结果 (此时背景会有噪声)
result = pipe.optimize(
    lr=0.05,
    batch_size=1,
    iters=1,
    width=width,
    height=height,
    weight=0.25,
    controller=controller,
    style_image=style_image,
    content_image=content_image,
    mixed_precision="no",
    num_inference_steps=num_inference_steps,
    enable_gradient_checkpoint=False,
)

# ==========================================
# 3. 掩膜后处理 (消除泄露)
# ==========================================
if os.path.exists(MASK_IMAGE_PATH):
    print(f"\n✂️ 正在应用掩膜切割背景...")
    
    # A. 读取掩膜
    mask_pil = Image.open(MASK_IMAGE_PATH).convert("L")
    mask_pil = mask_pil.resize((width, height), Image.LANCZOS)
    
    # B. 转为 Tensor (0=黑, 1=白)
    mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
    mask_tensor = mask_tensor.to(result.device)
    
    # C. 调整维度以匹配图像 [1, 3, H, W]
    if len(mask_tensor.shape) == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    elif len(mask_tensor.shape) == 3:
        mask_tensor = mask_tensor.unsqueeze(0).repeat(1, 3, 1, 1)

    # D. 可选：边缘羽化 (让切割边缘更自然)
    mask_tensor = F_t.gaussian_blur(mask_tensor, kernel_size=3, sigma=0.5)

    # E. 核心操作：哈达玛积 (Pixel-wise Multiply)
    # 结果 = 生成图 × 掩膜
    # 扇形内 × 1 = 保留；扇形外 × 0 = 消失
    result = result * mask_tensor
    
    print("✅ 背景已被置黑，风格泄露已消除")
else:
    print("⚠️ 未找到掩膜文件，结果将保留背景噪声！")

# ==========================================
# 4. 保存
# ==========================================
save_image(result, final_save_path)
print(f"\n🎉 处理完成！结果已保存: {final_save_path}")
