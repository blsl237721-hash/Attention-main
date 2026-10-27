import os
import time
import torch
import numpy as np
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
# 使用标准的 Diffusers UNet，而不是你自己改的那个
from diffusers import UNet2DConditionModel 
from pipeline_sd import ADPipeline
from utils import *

# ================= 🔧 配置区域 =================

# 1. 官方底模路径 (这就是最普通的 SD 1.5)
# 如果你本地有 runwayml/stable-diffusion-v1-5 就填本地路径
# 如果没有，填 "runwayml/stable-diffusion-v1-5" 让它自动下载
BASE_MODEL_PATH = "./models/stable-diffusion-v1-5"

# 2. 输入图片路径
STYLE_IMAGE_PATH = "./aircraft_11.jpg"   
CONTENT_IMAGE_PATH = "./marine-debris-aris3k-9.png"      

# 3. 输出设置
OUTPUT_DIR = "./standard_style_transfer"
OUTPUT_FILENAME = "result_standard.png"

# 4. 超参数 (默认参数)
width = 512
height = 512
seed = 42
num_inference_steps = 200
start_layer, end_layer = 10, 16  # Attention 注入层数

# ========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
final_save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

print(f"=== 标准 Attention 风格迁移 (使用原生 SD 1.5) ===")
print(f"风格: {STYLE_IMAGE_PATH}")
print(f"内容: {CONTENT_IMAGE_PATH}")

# ==========================================
# 1. 加载原生模型
# ==========================================
print("\n🔄 加载官方 SD 1.5 模型...")

# 加载 Scheduler
scheduler = DDIMScheduler.from_pretrained(BASE_MODEL_PATH, subfolder="scheduler")

# 加载 VAE
vae_model = AutoencoderKL.from_pretrained(BASE_MODEL_PATH, subfolder="vae")

# 【核心修改】加载官方标准的 UNet
# 这里没有任何 QuARF，也没有任何微调权重，就是纯净的 SD 1.5 UNet
print(f"👉 加载原生 UNet from: {BASE_MODEL_PATH}")
unet = UNet2DConditionModel.from_pretrained(
    BASE_MODEL_PATH, 
    subfolder="unet", 
    torch_dtype=torch.float16
)

# 组装 Pipeline (使用该仓库提供的 ADPipeline 进行 QKV 注入)
pipe = ADPipeline.from_pretrained(
    BASE_MODEL_PATH,
    unet=unet,
    scheduler=scheduler, 
    vae=vae_model,
    safety_checker=None, 
    torch_dtype=torch.float16
)

pipe.classifier = pipe.unet
pipe.to("cuda")
set_seed(seed)

# ==========================================
# 2. 执行推理
# ==========================================
if not os.path.exists(STYLE_IMAGE_PATH) or not os.path.exists(CONTENT_IMAGE_PATH):
    raise FileNotFoundError("找不到输入图片，请检查路径！")

style_image = load_image(STYLE_IMAGE_PATH, size=(width, height))
content_image = load_image(CONTENT_IMAGE_PATH, size=(width, height))

# 初始化 Attention 控制器
controller = Controller(self_layers=(start_layer, end_layer))

print("🚀 开始生成...")
start_time = time.time()

# 使用 ADPipeline 的 optimize 方法进行风格迁移
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
# 3. 保存结果
# ==========================================
save_image(result, final_save_path)

print(f"\n🎉 完成！耗时: {time.time() - start_time:.2f} 秒")
print(f"📁 结果已保存: {final_save_path}")
