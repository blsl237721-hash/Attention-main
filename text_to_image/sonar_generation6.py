from diffusers import StableDiffusionXLPipeline
import torch
from pytorch_wavelets import DWTForward

# 初始化流程保持不变
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    ignore_mismatched_sizes=True 
)

def materialize_down_wt_with_to_empty(down_wt_module, device="cuda", dtype=torch.float16):
    """为down_wt模块中的meta tensors分配内存"""
    print("开始处理down_wt模块的meta tensors...")
    
    # 移动到目标设备并分配内存
    down_wt_module.to_empty(device=device)
    
    # 确保所有参数和缓冲区都在正确设备上
    for name, param in down_wt_module.named_parameters():
        if param.is_meta:
            param.data = torch.empty_like(param.data, device=device, dtype=dtype)
            print(f"[Parameter {name}] 从meta device移动到 {device}")
        else:
            param.data = param.data.to(device=device, dtype=dtype)
            print(f"[Parameter {name}] 移动到 {device}")
    
    for name, buffer in down_wt_module.named_buffers():
        if buffer.is_meta:
            buffer.data = torch.empty_like(buffer.data, device=device, dtype=dtype)
            print(f"[Buffer {name}] 从meta device移动到 {device}")
        else:
            buffer.data = buffer.data.to(device=device, dtype=dtype)
            print(f"[Buffer {name}] 移动到 {device}")
    
    # 最终确认所有参数都在正确设备上
    down_wt_module.to(device=device, dtype=dtype)
    print("down_wt模块处理完成")
    return down_wt_module

# 处理UNet的down_wt模块
if hasattr(pipe.unet, 'down_wt'):
    pipe.unet.down_wt = materialize_down_wt_with_to_empty(pipe.unet.down_wt, device="cuda", dtype=torch.float16)
else:
    print("警告：未找到down_wt模块")



pipe = pipe.to("cuda", torch.float16)
pipe.load_lora_weights("./sdxl-lora-ew")
if hasattr(pipe.unet, 'down_wt') and hasattr(pipe.unet.down_wt, 'wt'):
    # 从 UNet 父模块获取设备和数据类型，而不是从 DWTForward
    device = pipe.unet.device
    dtype = pipe.unet.dtype
    
    print("重新初始化DWTForward模块以确保滤波器正确...")

    # 1. 创建一个新的 DWTForward 实例
    new_wt = DWTForward(J=1, mode='zero', wave='haar')

    # 2. 将新实例的内部张量手动移动到正确的设备和数据类型
    # DWTForward 内部的滤波器被注册为缓冲区
    for name, buffer in new_wt.named_buffers():
        setattr(new_wt, name, buffer.to(device, dtype))
        
    # 3. 将新实例赋值给模型
    pipe.unet.down_wt.wt = new_wt
    print("DWTForward模块重新初始化完成。")

prompt = "Sonar-detected wreckage of an undersea airplane,the fuselage broke off and the fuselage was partially covered by mud and sand."  
# 生成四张图像
images = pipe(
    prompt=prompt,
    negative_prompt="blurry, cartoonish, lowres.",
    num_inference_steps=30,
    guidance_scale=7.5,
    height=1024,
    width=1024,
    num_images_per_prompt=4  # 关键修改：指定生成数量
).images

# 保存所有生成的图像
for i, image in enumerate(images):
    image.save(f"output_{i}.png")
