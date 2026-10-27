import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import os
import glob

# ====================
# 函数和类定义 (保持不变)
# ====================

_gauss_kernel_cache = None

def gauss_kernel(channels=3, cuda=True):
    global _gauss_kernel_cache
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    
    _gauss_kernel_cache = kernel
    return  _gauss_kernel_cache.to('cuda')


def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = img.to(kernel.device)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def make_laplace(img, channels):
    img = img.to('cuda')
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff

def make_laplace_pyramid(img, level, channels):
    img = img.to('cuda')
    img = (img - img.min()) / (img.max() - img.min())
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = torch.abs(current - up)
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.cbam = CBAM(in_channels)
    def forward(self, edge_feature, x, pred):
        residual = x
        xsize = x.size()[2:]
        pred = torch.sigmoid(pred)
        background_att = (1 - pred).to('cuda')
        background_x= x * background_att
        edge_pred = make_laplace(pred, 1)
        pred_feature = x * edge_pred
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input
        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)
        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map
        out = fusion_feature + residual
        out = self.cbam(out)
        return out

def process_image(image_path, output_dir, level=4):
    """处理单个图像并保存高频特征"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        transform = transforms.ToTensor()
        img_tensor = transform(image).unsqueeze(0).to('cuda')
        
        # 生成拉普拉斯金字塔
        channels = img_tensor.shape[1]
        laplacian_pyramid = make_laplace_pyramid(img_tensor, level, channels)
        
        # 提取高频特征 (最精细的细节层)
        high_freq_feature = laplacian_pyramid[0]
        
        # 将多通道转换为单通道灰度图以便于保存
        if high_freq_feature.shape[1] > 1:
            gray_high_freq = torch.mean(high_freq_feature, dim=1, keepdim=True)
        else:
            gray_high_freq = high_freq_feature
        
        # 生成输出文件名（保持相同名称，只修改后缀为.png）
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.png")
        
        # 保存高频特征图
        torchvision.utils.save_image(gray_high_freq, output_path, normalize=True)
        
        print(f"已处理: {os.path.basename(image_path)} -> {os.path.basename(output_path)}")
        return True
        
    except Exception as e:
        print(f"处理图像 {os.path.basename(image_path)} 时出错: {str(e)}")
        return False

# ====================
# 主执行逻辑
# ====================
if __name__ == '__main__':
    # 设置输入和输出文件夹路径
    input_folder = './image2/'  # 输入文件夹路径
    output_folder = './image2/'  # 输出文件夹路径（与输入文件夹相同）
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    
    # 获取所有支持的图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, extension)))
        image_paths.extend(glob.glob(os.path.join(input_folder, extension.upper())))
    
    if not image_paths:
        print(f"在文件夹 {input_folder} 中未找到图像文件")
        exit()
    
    print(f"找到 {len(image_paths)} 个图像文件")
    
    # 处理每个图像
    success_count = 0
    for image_path in image_paths:
        if process_image(image_path, output_folder):
            success_count += 1
    
    print(f"\n处理完成! 成功处理 {success_count}/{len(image_paths)} 个图像")
    print(f"高频特征图已保存到: {output_folder}，文件名与原始图像相同，后缀为.png")
