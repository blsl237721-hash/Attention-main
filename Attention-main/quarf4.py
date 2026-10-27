import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, num_groups, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_groups = num_groups
        self.route_linear = nn.Linear(n_embed, num_experts * num_groups)
        self.noise_linear = nn.Linear(n_embed, num_experts * num_groups)

    def forward(self, mh_output):
        with torch.amp.autocast('cuda', enabled=False):
            mh_output = mh_output.float()
            mh_output = torch.clamp(mh_output, min=-100.0, max=100.0)

            batch_size = mh_output.shape[0]
            logits = self.route_linear(mh_output)
            noise_logits = self.noise_linear(mh_output)
            
            logits = torch.clamp(logits, min=-30.0, max=30.0)
            noise_logits = torch.clamp(noise_logits, min=-30.0, max=30.0)

            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            noisy_logits = logits + noise
            
            noisy_logits = noisy_logits.view(batch_size, self.num_groups, self.num_experts)
            top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
            zeros = torch.full_like(noisy_logits, float('-inf'))
            sparse_logits = zeros.scatter(-1, indices, top_k_logits)
            
            router_output = F.softmax(sparse_logits, dim=-1)
            return router_output, indices

class QuARF(nn.Module):
    def __init__(self, in_channels, quality_dim=4096, num_experts=4, top_k=4, num_groups=4, mode='soft'):
        super(QuARF, self).__init__()
        self.router = NoisyTopkRouter(quality_dim, num_experts, num_groups, top_k)
        self.num_groups = num_groups
        self.channels_per_group = in_channels // num_groups
        self.top_k = top_k
        self.mode = mode
        self.num_experts = num_experts

        kernel_size = 3
        self.experts = nn.ModuleList([
            nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size, padding=1, dilation=1),
            nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size, padding=2, dilation=2),
            nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size, padding=4, dilation=4),
            nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size, padding=8, dilation=8)
        ])

        # 零初始化
        for expert in self.experts:
            nn.init.zeros_(expert.weight)
            if expert.bias is not None:
                nn.init.zeros_(expert.bias)
        
        nn.init.normal_(self.router.route_linear.weight, std=0.01)
        nn.init.zeros_(self.router.route_linear.bias)

    def forward(self, x, quality_feature):
        original_dtype = x.dtype
        
        # ==================== 【新增】自动对齐 Batch Size ====================
        # 解决 RuntimeError: shape '[2, 1, 1, 1]' is invalid for input of size 4
        # 如果 quality_feature (B=4) 比 x (B=2) 多，直接截取前 B 个。
        if quality_feature.shape[0] != x.shape[0]:
            # print(f"⚠️ Warning: Batch mismatch! x: {x.shape[0]}, quality: {quality_feature.shape[0]}. Aligning...")
            if quality_feature.shape[0] > x.shape[0]:
                quality_feature = quality_feature[:x.shape[0]]
            else:
                raise RuntimeError(f"QuARF Batch Error: Quality features ({quality_feature.shape[0]}) < Images ({x.shape[0]}). Cannot proceed.")
        # ===================================================================

        # 全流程 FP32 保护
        with torch.amp.autocast('cuda', enabled=False):
            x_32 = x.float()
            f_32 = quality_feature.float()
            
            router_output, indices = self.router(f_32)  
            x_groups = x_32.chunk(self.num_groups, dim=1)
            outputs = []
            
            for group_idx, x_group in enumerate(x_groups):
                if self.mode == 'hard':
                    expert_indices = indices[:, group_idx, 0]  
                    group_output = torch.zeros_like(x_group)
                    for i in range(x_32.shape[0]):
                        expert = self.experts[expert_indices[i]]
                        group_output[i:i + 1] = expert(x_group[i:i + 1])
                elif self.mode == 'soft':
                    group_output = torch.zeros_like(x_group)
                    for i in range(self.num_experts):
                        expert = self.experts[i]
                        
                        # 使用 -1 自动适应 Batch Size，确保维度匹配
                        gating_scores = router_output[:, group_idx, i].view(-1, 1, 1, 1)
                        
                        expert_output = expert(x_group)
                        group_output += gating_scores * expert_output
                outputs.append(group_output)
            
            quarf_output_32 = torch.cat(outputs, dim=1)

        return x + quarf_output_32.to(original_dtype)