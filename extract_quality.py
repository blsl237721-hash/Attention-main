import os
import torch
import torchvision
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from CONTRIQUE.modules.CONTRIQUE_model import CONTRIQUE_model


# python extract_quality.py   --root_dir /home/ubuntu/project/DDPM/data/MarinepulsetoDDPMxunlian --output_dir features --model_path ~/project/DDPM/examples/models/CONTRIQUE_checkpoint25.tar

# ==== 图像预处理 ====
normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform_basic = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])

transform_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1),
    torchvision.transforms.ToTensor(),
    normalize
])


def process_image(img_path, model, device):
    img = Image.open(img_path).convert("RGB")

    image = transform_basic(img).unsqueeze(0).to(device)
    image_2 = transform_aug(img).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, _, _, model_feat, model_feat_2, _, _ = model(image, image_2)

    quality_feat = np.hstack((
        model_feat.detach().cpu().numpy(),
        model_feat_2.detach().cpu().numpy()
    ))
    quality_feat = torch.from_numpy(quality_feat).float().to(device)
    return quality_feat


def process_folder(root_dir, model, device):
    """
    Processes all images in subfolders and returns a dict mapping absolute image path to its feature.
    """
    all_quality_map = {}
    
    # 遍历 root_dir 下的所有内容
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        # 确保只处理子文件夹
        if not os.path.isdir(subdir_path):
            continue

        print(f"Processing directory: {subdir}")

        # 找到子文件夹中的所有图片文件
        files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 使用 tqdm 跟踪子文件夹进度
        # 注意：这里我们使用 files，而不是 sorted(os.listdir...)，以保持简洁，
        # 但如果您的原始数据是高度依赖 sorted 的，请保持 sorted(os.listdir(subdir_path))
        for filename in tqdm(files, desc=f"Extracting features for {subdir}"):
            img_path = os.path.join(subdir_path, filename)
            
            try:
                # feat 的形状为 (1, 4096)
                feat = process_image(img_path, model, device)
                
                # [关键修改] 将完整绝对路径作为键，特征张量（去除批次维度）作为值
                all_quality_map[os.path.abspath(img_path)] = feat.squeeze(0).cpu() 
                
            except Exception as e:
                print(f"\nSkipping {img_path}, error: {e}")
                continue
                
    return all_quality_map


def main():
    parser = argparse.ArgumentParser(description="Extract CONTRIQUE quality features from dataset folders")
    parser.add_argument("--root_dir", type=str, required=True, help="Input dataset directory (with subfolders)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save features")
    parser.add_argument("--model_path", type=str, default="./CONTRIQUE_weights.pth", help="Path to CONTRIQUE model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== 初始化 CONTRIQUE 模型 ====
    quality_model = CONTRIQUE_model(
    args=None,
    encoder=torchvision.models.resnet50(weights=None),
    n_features=2048)
    quality_model.load_state_dict(torch.load(args.model_path, map_location=device.type))
    quality_model.to(device)
    quality_model.eval()

    # ==== 遍历并处理 (调用修改后的函数) ====
    # 不再传入 output_dir
    all_quality_map = process_folder(args.root_dir, quality_model, device)
    
    # ==== 最终保存 ====
    if all_quality_map:
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, "all_quality_map.pt")
        torch.save(all_quality_map, save_path)
        print(f"\n=======================================================")
        print(f"✅ Extraction Complete. Saved all features to: {save_path}")
        print(f"Total images processed: {len(all_quality_map)}")
        print(f"=======================================================")




if __name__ == "__main__":
    main()
