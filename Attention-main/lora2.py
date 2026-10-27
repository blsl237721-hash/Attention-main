import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
from pipeline_sdxl import ADPipeline
from diffusers import DDIMScheduler, AutoencoderKL, StableDiffusionXLPipeline
import torch
from utils import *
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, LoraModel

# 模型路径和LoRA权重路径
model_name = os.path.expanduser("~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b")
lora_weights_path = "/home/xyj/Data/diffusers-main/examples/text_to_image/sdxl-lora-three/pytorch_lora_weights.safetensors"

# 其他参数
vae = ""
lr = 0.02
iters = 3
#seed = 42
mixed_precision = "bf16"
num_inference_steps = 50
guidance_scale = 7
num_images_per_prompt = 5
enable_gradient_checkpoint = True
start_layer, end_layer = 64, 70

seed = random.randint(0, 2**32 - 1)  # 生成0到2^32-1之间的随机整数
set_seed(seed)
print(f"Using random seed: {seed}")

prompt = "Sonar detected the wreckage of the shipwreck on the seafloor, with the hull relatively intact and covered by a thin layer of sediment, the shadow of which was clearly visible."
style_image = ["./data/style/shipwreck_112.jpg"]

# 加载调度器
scheduler = DDIMScheduler.from_pretrained(
    model_name, 
    subfolder="scheduler",
    local_files_only=True
)

pipe = ADPipeline.from_pretrained(
    model_name, 
    scheduler=scheduler, 
    safety_checker=None,
    local_files_only=True
)

# 加载LoRA权重
if os.path.exists(lora_weights_path):
    print(f"Loading LoRA weights from {lora_weights_path}")
    pipe.unet.load_attn_procs(lora_weights_path)
    print("LoRA weights loaded successfully!")
else:
    print(f"LoRA weights path {lora_weights_path} does not exist. Using base model only.")

# 加载VAE(如果有)
if vae != "":
    vae = AutoencoderKL.from_pretrained(vae, local_files_only=True)
    pipe.vae = vae

pipe.classifier = pipe.unet
set_seed(seed)

# 加载和处理风格图像
style_image = torch.cat([load_image(path, size=(1024, 1024)) for path in style_image])
controller = Controller(self_layers=(start_layer, end_layer))

# 生成图像
result = pipe.sample(
    controller=controller,
    iters=iters,
    lr=lr,
    adain=True,
    height=1024,
    width=1024,
    mixed_precision="bf16",
    style_image=style_image,
    prompt=prompt,
    negative_prompt="",
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    num_images_per_prompt=num_images_per_prompt,
    enable_gradient_checkpoint=enable_gradient_checkpoint
)

save_image(style_image, "style.png")

# 单独保存生成的每张图像
if result is not None:
    if len(result.shape) == 4:  # 批量图像 [B, C, H, W]
        for i in range(result.shape[0]):
            single_image = result[i]
            filename = f"output_{i}.png"
            save_image(single_image, filename)
            show_image(filename, title=f"{prompt} (Image {i+1})")
    else:  # 单张图像
        save_image(result, "output.png")
        show_image("output.png", title=prompt)
